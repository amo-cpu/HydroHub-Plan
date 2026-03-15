import streamlit as st
import pandas as pd
import numpy as np
import os, requests, zipfile, io
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="HydroHub AI")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

try:
    ORS_API_KEY = st.secrets["ORS_API_KEY"]
except Exception:
    ORS_API_KEY = None

# ──────────────────────────────────────────────────────────────
# ZIP-PREFIX LOOKUPS  (USPS public domain)
# Both tables keyed on the first 3 digits of the ZIP code.
# Covers every prefix used in the continental US + AK/HI.
# Used to fill City and State when the data source omits them.
# ──────────────────────────────────────────────────────────────
ZIP3_STATE = {
    "005":"NY","006":"PR","007":"PR","008":"VI","009":"PR",
    "010":"MA","011":"MA","012":"MA","013":"MA","014":"MA","015":"MA","016":"MA","017":"MA","018":"MA","019":"MA",
    "020":"MA","021":"MA","022":"MA","023":"MA","024":"MA","025":"MA","026":"MA","027":"MA",
    "028":"RI","029":"RI","030":"NH","031":"NH","032":"NH","033":"NH","034":"NH","035":"NH","036":"NH","037":"NH","038":"NH",
    "039":"ME","040":"ME","041":"ME","042":"ME","043":"ME","044":"ME","045":"ME","046":"ME","047":"ME","048":"ME","049":"ME",
    "050":"VT","051":"VT","052":"VT","053":"VT","054":"VT","055":"VT","056":"VT","057":"VT","058":"VT","059":"VT",
    "060":"CT","061":"CT","062":"CT","063":"CT","064":"CT","065":"CT","066":"CT","067":"CT","068":"CT","069":"CT",
    "070":"NJ","071":"NJ","072":"NJ","073":"NJ","074":"NJ","075":"NJ","076":"NJ","077":"NJ","078":"NJ","079":"NJ",
    "080":"NJ","081":"NJ","082":"NJ","083":"NJ","084":"NJ","085":"NJ","086":"NJ","087":"NJ","088":"NJ","089":"NJ",
    "100":"NY","101":"NY","102":"NY","103":"NY","104":"NY","105":"NY","106":"NY","107":"NY","108":"NY","109":"NY",
    "110":"NY","111":"NY","112":"NY","113":"NY","114":"NY","115":"NY","116":"NY","117":"NY","118":"NY","119":"NY",
    "120":"NY","121":"NY","122":"NY","123":"NY","124":"NY","125":"NY","126":"NY","127":"NY","128":"NY","129":"NY",
    "130":"NY","131":"NY","132":"NY","133":"NY","134":"NY","135":"NY","136":"NY","137":"NY","138":"NY","139":"NY",
    "140":"NY","141":"NY","142":"NY","143":"NY","144":"NY","145":"NY","146":"NY","147":"NY","148":"NY","149":"NY",
    "150":"PA","151":"PA","152":"PA","153":"PA","154":"PA","155":"PA","156":"PA","157":"PA","158":"PA","159":"PA",
    "160":"PA","161":"PA","162":"PA","163":"PA","164":"PA","165":"PA","166":"PA","167":"PA","168":"PA","169":"PA",
    "170":"PA","171":"PA","172":"PA","173":"PA","174":"PA","175":"PA","176":"PA","177":"PA","178":"PA","179":"PA",
    "180":"PA","181":"PA","182":"PA","183":"PA","184":"PA","185":"PA","186":"PA","187":"PA","188":"PA","189":"PA",
    "190":"PA","191":"PA","192":"PA","193":"PA","194":"PA","195":"PA","196":"PA",
    "197":"DE","198":"DE","199":"DE","200":"DC","201":"VA","202":"DC","203":"DC","204":"DC","205":"DC",
    "206":"MD","207":"MD","208":"MD","209":"MD","210":"MD","211":"MD","212":"MD","214":"MD","215":"MD",
    "216":"MD","217":"MD","218":"MD","219":"MD",
    "220":"VA","221":"VA","222":"VA","223":"VA","224":"VA","225":"VA","226":"VA","227":"VA","228":"VA","229":"VA",
    "230":"VA","231":"VA","232":"VA","233":"VA","234":"VA","235":"VA","236":"VA","237":"VA","238":"VA","239":"VA",
    "240":"VA","241":"VA","242":"VA","243":"VA","244":"VA","245":"VA","246":"VA",
    "247":"WV","248":"WV","249":"WV","250":"WV","251":"WV","252":"WV","253":"WV","254":"WV","255":"WV",
    "256":"WV","257":"WV","258":"WV","259":"WV","260":"WV","261":"WV","262":"WV","263":"WV","264":"WV",
    "265":"WV","266":"WV","267":"WV","268":"WV",
    "270":"NC","271":"NC","272":"NC","273":"NC","274":"NC","275":"NC","276":"NC","277":"NC","278":"NC","279":"NC",
    "280":"NC","281":"NC","282":"NC","283":"NC","284":"NC","285":"NC","286":"NC","287":"NC","288":"NC","289":"NC",
    "290":"SC","291":"SC","292":"SC","293":"SC","294":"SC","295":"SC","296":"SC","297":"SC","298":"SC","299":"SC",
    "300":"GA","301":"GA","302":"GA","303":"GA","304":"GA","305":"GA","306":"GA","307":"GA","308":"GA","309":"GA",
    "310":"GA","311":"GA","312":"GA","313":"GA","314":"GA","315":"GA","316":"GA","317":"GA","318":"GA","319":"GA",
    "320":"FL","321":"FL","322":"FL","323":"FL","324":"FL","325":"FL","326":"FL","327":"FL","328":"FL","329":"FL",
    "330":"FL","331":"FL","332":"FL","333":"FL","334":"FL","335":"FL","336":"FL","337":"FL","338":"FL","339":"FL",
    "340":"HI","341":"FL","342":"FL","344":"FL","346":"FL","347":"FL","349":"FL",
    "350":"AL","351":"AL","352":"AL","354":"AL","355":"AL","356":"AL","357":"AL","358":"AL","359":"AL",
    "360":"AL","361":"AL","362":"AL","363":"AL","364":"AL","365":"AL","366":"AL","367":"AL","368":"AL","369":"AL",
    "370":"TN","371":"TN","372":"TN","373":"TN","374":"TN","376":"TN","377":"TN","378":"TN","379":"TN",
    "380":"TN","381":"TN","382":"TN","383":"TN","384":"TN","385":"TN",
    "386":"MS","387":"MS","388":"MS","389":"MS","390":"MS","391":"MS","392":"MS","393":"MS","394":"MS","395":"MS","396":"MS","397":"MS",
    "398":"GA","399":"GA","400":"KY","401":"KY","402":"KY","403":"KY","404":"KY","405":"KY","406":"KY","407":"KY","408":"KY","409":"KY",
    "410":"KY","411":"KY","412":"KY","413":"KY","414":"KY","415":"KY","416":"KY","417":"KY","418":"KY",
    "420":"KY","421":"KY","422":"KY","423":"KY","424":"KY","425":"KY","426":"KY","427":"KY",
    "430":"OH","431":"OH","432":"OH","433":"OH","434":"OH","435":"OH","436":"OH","437":"OH","438":"OH","439":"OH",
    "440":"OH","441":"OH","442":"OH","443":"OH","444":"OH","445":"OH","446":"OH","447":"OH","448":"OH","449":"OH",
    "450":"OH","451":"OH","452":"OH","453":"OH","454":"OH","455":"OH","456":"OH","457":"OH","458":"OH",
    "460":"IN","461":"IN","462":"IN","463":"IN","464":"IN","465":"IN","466":"IN","467":"IN","468":"IN","469":"IN",
    "470":"IN","471":"IN","472":"IN","473":"IN","474":"IN","475":"IN","476":"IN","477":"IN","478":"IN","479":"IN",
    "480":"MI","481":"MI","482":"MI","483":"MI","484":"MI","485":"MI","486":"MI","487":"MI","488":"MI","489":"MI",
    "490":"MI","491":"MI","492":"MI","493":"MI","494":"MI","495":"MI","496":"MI","497":"MI","498":"MI","499":"MI",
    "500":"IA","501":"IA","502":"IA","503":"IA","504":"IA","505":"IA","506":"IA","507":"IA","508":"IA","509":"IA",
    "510":"IA","511":"IA","512":"IA","513":"IA","514":"IA","515":"IA","516":"IA","520":"IA","521":"IA","522":"IA",
    "523":"IA","524":"IA","525":"IA","526":"IA","527":"IA","528":"IA",
    "530":"WI","531":"WI","532":"WI","534":"WI","535":"WI","537":"WI","538":"WI","539":"WI","540":"WI","541":"WI",
    "542":"WI","543":"WI","544":"WI","545":"WI","546":"WI","547":"WI","548":"WI","549":"WI",
    "550":"MN","551":"MN","553":"MN","554":"MN","555":"MN","556":"MN","557":"MN","558":"MN","559":"MN",
    "560":"MN","561":"MN","562":"MN","563":"MN","564":"MN","565":"MN","566":"MN","567":"MN",
    "570":"SD","571":"SD","572":"SD","573":"SD","574":"SD","575":"SD","576":"SD","577":"SD",
    "580":"ND","581":"ND","582":"ND","583":"ND","584":"ND","585":"ND","586":"ND","587":"ND","588":"ND",
    "590":"MT","591":"MT","592":"MT","593":"MT","594":"MT","595":"MT","596":"MT","597":"MT","598":"MT","599":"MT",
    "600":"IL","601":"IL","602":"IL","603":"IL","604":"IL","605":"IL","606":"IL","607":"IL","608":"IL","609":"IL",
    "610":"IL","611":"IL","612":"IL","613":"IL","614":"IL","615":"IL","616":"IL","617":"IL","618":"IL","619":"IL",
    "620":"IL","621":"IL","622":"IL","623":"IL","624":"IL","625":"IL","626":"IL","627":"IL","628":"IL","629":"IL",
    "630":"MO","631":"MO","633":"MO","634":"MO","635":"MO","636":"MO","637":"MO","638":"MO","639":"MO",
    "640":"MO","641":"MO","644":"MO","645":"MO","646":"MO","647":"MO","648":"MO","649":"MO",
    "650":"MO","651":"MO","652":"MO","653":"MO","654":"MO","655":"MO","656":"MO","657":"MO","658":"MO",
    "660":"KS","661":"KS","662":"KS","664":"KS","665":"KS","666":"KS","667":"KS","668":"KS","669":"KS",
    "670":"KS","671":"KS","672":"KS","673":"KS","674":"KS","675":"KS","676":"KS","677":"KS","678":"KS","679":"KS",
    "680":"NE","681":"NE","683":"NE","684":"NE","685":"NE","686":"NE","687":"NE","688":"NE","689":"NE",
    "690":"NE","691":"NE","692":"NE","693":"NE","700":"LA","701":"LA","703":"LA","704":"LA","705":"LA",
    "706":"LA","707":"LA","708":"LA","710":"LA","711":"LA","712":"LA","713":"LA","714":"LA",
    "716":"AR","717":"AR","718":"AR","719":"AR","720":"AR","721":"AR","722":"AR","723":"AR","724":"AR",
    "725":"AR","726":"AR","727":"AR","728":"AR","729":"AR",
    "730":"OK","731":"OK","733":"OK","734":"OK","735":"OK","736":"OK","737":"OK","738":"OK","739":"OK",
    "740":"OK","741":"OK","743":"OK","744":"OK","745":"OK","746":"OK","747":"OK","748":"OK","749":"OK",
    "750":"TX","751":"TX","752":"TX","753":"TX","754":"TX","755":"TX","756":"TX","757":"TX","758":"TX","759":"TX",
    "760":"TX","761":"TX","762":"TX","763":"TX","764":"TX","765":"TX","766":"TX","767":"TX","768":"TX","769":"TX",
    "770":"TX","771":"TX","772":"TX","773":"TX","774":"TX","775":"TX","776":"TX","777":"TX","778":"TX","779":"TX",
    "780":"TX","781":"TX","782":"TX","783":"TX","784":"TX","785":"TX","786":"TX","787":"TX","788":"TX","789":"TX",
    "790":"TX","791":"TX","792":"TX","793":"TX","794":"TX","795":"TX","796":"TX","797":"TX","798":"TX","799":"TX",
    "800":"CO","801":"CO","802":"CO","803":"CO","804":"CO","805":"CO","806":"CO","807":"CO","808":"CO","809":"CO",
    "810":"CO","811":"CO","812":"CO","813":"CO","814":"CO","815":"CO","816":"CO",
    "820":"WY","821":"WY","822":"WY","823":"WY","824":"WY","825":"WY","826":"WY","827":"WY","828":"WY",
    "829":"WY","830":"WY","831":"WY","832":"ID","833":"ID","834":"ID","835":"ID","836":"ID","837":"ID","838":"ID",
    "840":"UT","841":"UT","842":"UT","843":"UT","844":"UT","845":"UT","846":"UT","847":"UT",
    "850":"AZ","851":"AZ","852":"AZ","853":"AZ","855":"AZ","856":"AZ","857":"AZ","859":"AZ",
    "860":"AZ","863":"AZ","864":"AZ","865":"AZ",
    "870":"NM","871":"NM","872":"NM","873":"NM","874":"NM","875":"NM","876":"NM","877":"NM","878":"NM",
    "879":"NM","880":"NM","881":"NM","882":"NM","883":"NM","884":"NM","885":"TX",
    "889":"NV","890":"NV","891":"NV","893":"NV","894":"NV","895":"NV","897":"NV","898":"NV",
    "900":"CA","901":"CA","902":"CA","903":"CA","904":"CA","905":"CA","906":"CA","907":"CA","908":"CA",
    "910":"CA","911":"CA","912":"CA","913":"CA","914":"CA","915":"CA","916":"CA","917":"CA","918":"CA","919":"CA",
    "920":"CA","921":"CA","922":"CA","923":"CA","924":"CA","925":"CA","926":"CA","927":"CA","928":"CA",
    "930":"CA","931":"CA","932":"CA","933":"CA","934":"CA","935":"CA","936":"CA","937":"CA","938":"CA","939":"CA",
    "940":"CA","941":"CA","943":"CA","944":"CA","945":"CA","946":"CA","947":"CA","948":"CA","949":"CA",
    "950":"CA","951":"CA","952":"CA","953":"CA","954":"CA","955":"CA","956":"CA","957":"CA","958":"CA","959":"CA",
    "960":"CA","961":"CA","967":"HI","968":"HI",
    "970":"OR","971":"OR","972":"OR","973":"OR","974":"OR","975":"OR","976":"OR","977":"OR","978":"OR","979":"OR",
    "980":"WA","981":"WA","982":"WA","983":"WA","984":"WA","985":"WA","986":"WA","988":"WA","989":"WA",
    "990":"WA","991":"WA","992":"WA","993":"WA","994":"WA",
    "995":"AK","996":"AK","997":"AK","998":"AK","999":"AK",
}

ZIP3_CITY = {
    "010":"Springfield","011":"Springfield","012":"Pittsfield","013":"Greenfield",
    "014":"Fitchburg","015":"Worcester","016":"Worcester","017":"Worcester",
    "018":"Lowell","019":"Lynn","020":"Boston","021":"Boston","022":"Boston",
    "023":"Brockton","024":"Norwood","025":"Hyannis","026":"Hyannis","027":"Hyannis",
    "028":"Providence","029":"Newport","030":"Manchester","031":"Manchester",
    "032":"Concord","033":"Concord","034":"Keene","035":"Claremont",
    "036":"Portsmouth","037":"Portsmouth","038":"Portsmouth",
    "039":"Portland","040":"Portland","041":"Portland","042":"Portland",
    "043":"Augusta","044":"Bangor","045":"Bangor","046":"Waterville",
    "047":"Rockland","048":"Bath","049":"Lewiston",
    "050":"Burlington","051":"Burlington","052":"Burlington","053":"Rutland",
    "054":"Burlington","055":"Burlington","056":"Burlington","057":"Montpelier",
    "058":"Newport","059":"Lyndonville",
    "060":"Hartford","061":"Hartford","062":"Willimantic","063":"New London",
    "064":"New Haven","065":"New Haven","066":"Bridgeport","067":"Danbury",
    "068":"Stamford","069":"Greenwich",
    "070":"Newark","071":"Newark","072":"Elizabeth","073":"Elizabeth",
    "074":"Paterson","075":"Paterson","076":"Hackensack","077":"Long Branch",
    "078":"Dover","079":"Morristown",
    "080":"Trenton","081":"Trenton","082":"Atlantic City","083":"Vineland",
    "084":"Atlantic City","085":"Trenton","086":"Trenton","087":"Lakewood",
    "088":"New Brunswick","089":"New Brunswick",
    "100":"New York","101":"New York","102":"New York","103":"Staten Island",
    "104":"Bronx","105":"Westchester","106":"White Plains","107":"Yonkers",
    "108":"New Rochelle","109":"Suffern",
    "110":"Queens","111":"Queens","112":"Brooklyn","113":"Queens",
    "114":"Queens","115":"Jamaica","116":"Far Rockaway","117":"Hempstead",
    "118":"Hicksville","119":"Babylon",
    "120":"Albany","121":"Albany","122":"Albany","123":"Schenectady",
    "124":"Kingston","125":"Poughkeepsie","126":"Middletown","127":"Newburgh",
    "128":"Binghamton","129":"Oneonta",
    "130":"Syracuse","131":"Syracuse","132":"Syracuse","133":"Utica",
    "134":"Utica","135":"Watertown","136":"Watertown","137":"Elmira",
    "138":"Elmira","139":"Elmira",
    "140":"Buffalo","141":"Buffalo","142":"Buffalo","143":"Niagara Falls",
    "144":"Rochester","145":"Rochester","146":"Rochester","147":"Corning",
    "148":"Ithaca","149":"Elmira",
    "150":"Pittsburgh","151":"Pittsburgh","152":"Pittsburgh","153":"Pittsburgh",
    "154":"Pittsburgh","155":"Uniontown","156":"Greensburg","157":"Indiana",
    "158":"Indiana","159":"Kittanning",
    "160":"Butler","161":"New Castle","162":"New Castle","163":"New Castle",
    "164":"Erie","165":"Erie","166":"Erie","167":"Clarion",
    "168":"Lewisburg","169":"Williamsport",
    "170":"Harrisburg","171":"Harrisburg","172":"Harrisburg","173":"York",
    "174":"York","175":"Lancaster","176":"Lancaster","177":"Lancaster",
    "178":"Sunbury","179":"Sunbury",
    "180":"Allentown","181":"Allentown","182":"Allentown","183":"Stroudsburg",
    "184":"Scranton","185":"Scranton","186":"Scranton","187":"Wilkes-Barre",
    "188":"Wilkes-Barre","189":"Hazleton",
    "190":"Philadelphia","191":"Philadelphia","192":"Philadelphia","193":"Chester",
    "194":"Norristown","195":"Reading","196":"Reading",
    "197":"Wilmington","198":"Wilmington","199":"Wilmington",
    "200":"Washington DC","201":"Arlington","202":"Washington DC",
    "203":"Washington DC","204":"Washington DC","205":"Washington DC",
    "206":"Rockville","207":"Rockville","208":"Bethesda","209":"Silver Spring",
    "210":"Baltimore","211":"Baltimore","212":"Baltimore","214":"Annapolis",
    "215":"Cumberland","216":"Hagerstown","217":"Frederick","218":"Salisbury","219":"Salisbury",
    "220":"Arlington","221":"Alexandria","222":"Arlington","223":"Arlington",
    "224":"Fredericksburg","225":"Fredericksburg","226":"Charlottesville",
    "227":"Harrisonburg","228":"Staunton","229":"Staunton",
    "230":"Richmond","231":"Richmond","232":"Richmond","233":"Norfolk",
    "234":"Norfolk","235":"Norfolk","236":"Norfolk","237":"Portsmouth",
    "238":"Newport News","239":"Hampton",
    "240":"Roanoke","241":"Roanoke","242":"Bristol","243":"Bluefield",
    "244":"Lynchburg","245":"Lynchburg","246":"Danville",
    "247":"Huntington","248":"Charleston","249":"Charleston",
    "250":"Charleston","251":"Charleston","252":"Charleston","253":"Charleston",
    "254":"Charleston","255":"Huntington","256":"Huntington","257":"Parkersburg",
    "258":"Parkersburg","259":"Lewisburg","260":"Wheeling","261":"Wheeling",
    "262":"Morgantown","263":"Clarksburg","264":"Elkins","265":"Weston",
    "266":"Buckhannon","267":"Elkins","268":"Romney",
    "270":"Greensboro","271":"Winston-Salem","272":"Greensboro","273":"Greensboro",
    "274":"Greensboro","275":"Raleigh","276":"Raleigh","277":"Raleigh",
    "278":"Rocky Mount","279":"Rocky Mount",
    "280":"Charlotte","281":"Charlotte","282":"Charlotte","283":"Charlotte",
    "284":"Wilmington","285":"Fayetteville","286":"Asheville","287":"Asheville",
    "288":"Asheville","289":"Hickory",
    "290":"Columbia","291":"Columbia","292":"Columbia","293":"Spartanburg",
    "294":"Charleston","295":"Greenville","296":"Greenville","297":"Rock Hill",
    "298":"Augusta","299":"Beaufort",
    "300":"Atlanta","301":"Atlanta","302":"Atlanta","303":"Atlanta",
    "304":"Atlanta","305":"Atlanta","306":"Atlanta","307":"Dalton",
    "308":"Augusta","309":"Augusta",
    "310":"Macon","311":"Macon","312":"Savannah","313":"Savannah",
    "314":"Savannah","315":"Waycross","316":"Valdosta","317":"Albany",
    "318":"Columbus","319":"Albany",
    "320":"Jacksonville","321":"Daytona Beach","322":"Jacksonville",
    "323":"Tallahassee","324":"Gainesville","325":"Gainesville",
    "326":"Jacksonville","327":"Orlando","328":"Orlando","329":"Melbourne",
    "330":"Miami","331":"Miami","332":"Miami","333":"Fort Lauderdale",
    "334":"West Palm Beach","335":"Tampa","336":"Tampa","337":"Tampa",
    "338":"Lakeland","339":"Fort Myers","341":"Naples","342":"Sarasota",
    "344":"Fort Myers","346":"Clearwater","347":"Orlando","349":"Fort Pierce",
    "350":"Birmingham","351":"Birmingham","352":"Birmingham","354":"Birmingham",
    "355":"Birmingham","356":"Anniston","357":"Huntsville","358":"Huntsville",
    "359":"Florence","360":"Montgomery","361":"Mobile","362":"Mobile",
    "363":"Dothan","364":"Selma","365":"Mobile","366":"Mobile",
    "367":"Decatur","368":"Birmingham","369":"Gadsden",
    "370":"Nashville","371":"Nashville","372":"Nashville","373":"Chattanooga",
    "374":"Chattanooga","376":"Johnson City","377":"Knoxville","378":"Knoxville",
    "379":"Knoxville",
    "380":"Memphis","381":"Memphis","382":"Memphis","383":"Jackson",
    "384":"Jackson","385":"Jackson",
    "386":"Greenville","387":"Columbus","388":"Tupelo","389":"Meridian",
    "390":"Jackson","391":"Jackson","392":"Hattiesburg","393":"Biloxi",
    "394":"Gulfport","395":"Gulfport","396":"McComb",
    "400":"Louisville","401":"Louisville","402":"Louisville","403":"Lexington",
    "404":"Lexington","405":"Lexington","406":"Frankfort","407":"Elizabethtown",
    "408":"Bowling Green","409":"Bowling Green",
    "410":"Covington","411":"Ashland","412":"Pikeville","413":"Pikeville",
    "414":"Pikeville","415":"Hazard","416":"Hazard","417":"Corbin","418":"Corbin",
    "420":"Paducah","421":"Owensboro","422":"Owensboro","423":"Hopkinsville",
    "424":"Hopkinsville","425":"Somerset","426":"Somerset","427":"Danville",
    "430":"Columbus","431":"Columbus","432":"Columbus","433":"Marion",
    "434":"Toledo","435":"Toledo","436":"Toledo","437":"Zanesville",
    "438":"Zanesville","439":"Steubenville",
    "440":"Cleveland","441":"Cleveland","442":"Cleveland","443":"Elyria",
    "444":"Youngstown","445":"Youngstown","446":"Akron","447":"Akron",
    "448":"Mansfield","449":"Marion",
    "450":"Cincinnati","451":"Cincinnati","452":"Cincinnati","453":"Dayton",
    "454":"Dayton","455":"Springfield","456":"Columbus","457":"Athens","458":"Lima",
    "460":"Indianapolis","461":"Indianapolis","462":"Indianapolis",
    "463":"Gary","464":"Gary","465":"South Bend","466":"South Bend",
    "467":"Fort Wayne","468":"Fort Wayne","469":"Kokomo",
    "470":"Anderson","471":"Louisville","472":"Columbus","473":"Muncie",
    "474":"Bloomington","475":"Terre Haute","476":"Evansville",
    "477":"Evansville","478":"Evansville","479":"Lafayette",
    "480":"Detroit","481":"Detroit","482":"Detroit","483":"Ann Arbor",
    "484":"Flint","485":"Flint","486":"Saginaw","487":"Bay City",
    "488":"Lansing","489":"Lansing",
    "490":"Kalamazoo","491":"Kalamazoo","492":"Battle Creek","493":"Grand Rapids",
    "494":"Grand Rapids","495":"Muskegon","496":"Traverse City",
    "497":"Iron Mountain","498":"Marquette","499":"Sault Ste Marie",
    "500":"Des Moines","501":"Des Moines","502":"Des Moines",
    "503":"Des Moines","504":"Mason City","505":"Waterloo","506":"Waterloo",
    "507":"Dubuque","508":"Iowa City",
    "510":"Sioux City","511":"Sioux City","512":"Sioux City",
    "513":"Council Bluffs","514":"Carroll","515":"Des Moines","516":"Ottumwa",
    "520":"Davenport","521":"Davenport","522":"Davenport","523":"Davenport",
    "524":"Iowa City","525":"Burlington","526":"Burlington","527":"Keokuk","528":"Burlington",
    "530":"Milwaukee","531":"Milwaukee","532":"Milwaukee","534":"Racine",
    "535":"Kenosha","537":"Madison","538":"Madison","539":"Madison",
    "540":"Green Bay","541":"Green Bay","542":"Green Bay","543":"Sheboygan",
    "544":"Oshkosh","545":"Wausau","546":"La Crosse","547":"Eau Claire",
    "548":"Appleton","549":"Green Bay",
    "550":"Minneapolis","551":"St Paul","553":"Minneapolis","554":"Minneapolis",
    "555":"Minneapolis","556":"Duluth","557":"Duluth","558":"Brainerd",
    "559":"Rochester","560":"Mankato","561":"St Cloud","562":"St Cloud",
    "563":"St Cloud","564":"Bemidji","565":"Grand Forks","566":"Grand Forks","567":"Moorhead",
    "570":"Sioux Falls","571":"Sioux Falls","572":"Watertown","573":"Aberdeen",
    "574":"Rapid City","575":"Rapid City","576":"Mobridge","577":"Pierre",
    "580":"Fargo","581":"Fargo","582":"Grand Forks","583":"Minot",
    "584":"Minot","585":"Bismarck","586":"Bismarck","587":"Jamestown","588":"Williston",
    "590":"Billings","591":"Billings","592":"Havre","593":"Great Falls",
    "594":"Great Falls","595":"Helena","596":"Missoula","597":"Missoula",
    "598":"Lewistown","599":"Miles City",
    "600":"Chicago","601":"Chicago","602":"Chicago","603":"Chicago",
    "604":"Chicago","605":"Chicago","606":"Chicago","607":"Chicago",
    "608":"Joliet","609":"Kankakee",
    "610":"Rockford","611":"Rockford","612":"Peoria","613":"Peoria",
    "614":"Peoria","615":"Galesburg","616":"Bloomington","617":"Decatur",
    "618":"Springfield","619":"Springfield",
    "620":"East St Louis","621":"East St Louis","622":"Alton","623":"Quincy",
    "624":"Effingham","625":"Springfield","626":"Springfield","627":"Springfield",
    "628":"Mount Vernon","629":"Carbondale",
    "630":"St Louis","631":"St Louis","633":"St Louis","634":"St Louis",
    "635":"Hannibal","636":"Cape Girardeau","637":"Poplar Bluff",
    "638":"Sikeston","639":"Jefferson City",
    "640":"Kansas City","641":"Kansas City","644":"St Joseph","645":"St Joseph",
    "646":"Chillicothe","647":"Trenton","648":"Joplin","649":"Joplin",
    "650":"Columbia","651":"Columbia","652":"Columbia","653":"Rolla",
    "654":"Springfield","655":"Springfield","656":"Springfield",
    "657":"Springfield","658":"Springfield",
    "660":"Wichita","661":"Wichita","662":"Wichita","664":"Topeka","665":"Topeka",
    "666":"Manhattan","667":"Salina","668":"Hays","669":"Liberal",
    "670":"Wichita","671":"Wichita","672":"Wichita","673":"Hutchinson",
    "674":"Hutchinson","675":"Dodge City","676":"Dodge City",
    "677":"Garden City","678":"Liberal","679":"Emporia",
    "680":"Omaha","681":"Omaha","683":"Lincoln","684":"Lincoln","685":"Lincoln",
    "686":"Grand Island","687":"Grand Island","688":"Norfolk","689":"Fremont",
    "690":"McCook","691":"North Platte","692":"Scottsbluff","693":"Scottsbluff",
    "700":"New Orleans","701":"New Orleans","703":"New Orleans","704":"New Orleans",
    "705":"Lafayette","706":"Lake Charles","707":"Lake Charles","708":"Baton Rouge",
    "710":"Shreveport","711":"Shreveport","712":"Shreveport",
    "713":"Alexandria","714":"Alexandria",
    "716":"Pine Bluff","717":"Fort Smith","718":"Fort Smith","719":"Fort Smith",
    "720":"Little Rock","721":"Little Rock","722":"Little Rock",
    "723":"Jonesboro","724":"Jonesboro","725":"Batesville",
    "726":"Harrison","727":"Fayetteville","728":"Fayetteville","729":"Fort Smith",
    "730":"Oklahoma City","731":"Oklahoma City","733":"Ardmore",
    "734":"Lawton","735":"Lawton","736":"Enid","737":"Enid",
    "738":"Woodward","739":"Woodward",
    "740":"Tulsa","741":"Tulsa","743":"Muskogee","744":"McAlester",
    "745":"McAlester","746":"Ponca City","747":"Bartlesville",
    "748":"Miami","749":"Okmulgee",
    "750":"Dallas","751":"Dallas","752":"Dallas","753":"Dallas","754":"Dallas",
    "755":"Waxahachie","756":"Tyler","757":"Tyler","758":"Palestine",
    "759":"Longview",
    "760":"Fort Worth","761":"Fort Worth","762":"Fort Worth","763":"Weatherford",
    "764":"Stephenville","765":"Waco","766":"Waco","767":"Waco",
    "768":"Abilene","769":"San Angelo",
    "770":"Houston","771":"Houston","772":"Houston","773":"Houston",
    "774":"Houston","775":"Galveston","776":"Houston","777":"Houston",
    "778":"Beaumont","779":"Beaumont",
    "780":"San Antonio","781":"San Antonio","782":"San Antonio","783":"San Antonio",
    "784":"Laredo","785":"McAllen","786":"Austin","787":"Austin",
    "788":"Del Rio","789":"Uvalde",
    "790":"Amarillo","791":"Amarillo","792":"Wichita Falls","793":"Wichita Falls",
    "794":"San Angelo","795":"Lubbock","796":"Lubbock","797":"Midland",
    "798":"El Paso","799":"El Paso",
    "800":"Denver","801":"Denver","802":"Denver","803":"Aurora",
    "804":"Aurora","805":"Colorado Springs","806":"Colorado Springs",
    "807":"Colorado Springs","808":"Pueblo","809":"Pueblo",
    "810":"Alamosa","811":"Durango","812":"Durango","813":"Grand Junction",
    "814":"Grand Junction","815":"Fort Collins","816":"Fort Collins",
    "820":"Cheyenne","821":"Cheyenne","822":"Casper","823":"Casper",
    "824":"Gillette","825":"Riverton","826":"Cody","827":"Sheridan",
    "828":"Rock Springs","829":"Laramie","830":"Laramie","831":"Rock Springs",
    "832":"Boise","833":"Boise","834":"Twin Falls","835":"Twin Falls",
    "836":"Boise","837":"Pocatello","838":"Pocatello",
    "840":"Salt Lake City","841":"Salt Lake City","842":"Salt Lake City",
    "843":"Salt Lake City","844":"Ogden","845":"Provo","846":"Provo","847":"Provo",
    "850":"Phoenix","851":"Phoenix","852":"Phoenix","853":"Phoenix",
    "855":"Mesa","856":"Tucson","857":"Tucson","859":"Tucson",
    "860":"Flagstaff","863":"Prescott","864":"Yuma","865":"Albuquerque",
    "870":"Albuquerque","871":"Albuquerque","872":"Gallup",
    "873":"Las Vegas NM","874":"Santa Fe","875":"Albuquerque",
    "876":"Farmington","877":"Roswell","878":"Las Cruces","879":"Las Cruces",
    "880":"Las Cruces","881":"Las Cruces","882":"Deming",
    "883":"Alamogordo","884":"Carlsbad","885":"El Paso",
    "889":"Las Vegas","890":"Las Vegas","891":"Las Vegas",
    "893":"Reno","894":"Reno","895":"Reno","897":"Carson City","898":"Elko",
    "900":"Los Angeles","901":"Los Angeles","902":"Inglewood","903":"Torrance",
    "904":"Santa Monica","905":"Long Beach","906":"Long Beach","907":"Torrance",
    "908":"Long Beach","910":"Pasadena","911":"Pasadena","912":"Glendale",
    "913":"Burbank","914":"Van Nuys","915":"Canoga Park","916":"San Fernando",
    "917":"Alhambra","918":"El Monte","919":"San Bernardino",
    "920":"San Diego","921":"San Diego","922":"San Diego",
    "923":"San Diego","924":"San Diego",
    "925":"Riverside","926":"Anaheim","927":"Santa Ana","928":"Anaheim",
    "930":"Ventura","931":"Oxnard","932":"Bakersfield","933":"Bakersfield",
    "934":"Santa Barbara","935":"Santa Barbara","936":"Fresno","937":"Fresno",
    "938":"Fresno","939":"Salinas","940":"San Francisco","941":"San Francisco",
    "943":"Palo Alto","944":"Oakland","945":"Oakland","946":"Oakland",
    "947":"Berkeley","948":"Richmond","949":"Oakland",
    "950":"San Jose","951":"San Jose","952":"San Jose","953":"Santa Cruz",
    "954":"Santa Rosa","955":"Eureka","956":"Sacramento","957":"Sacramento",
    "958":"Sacramento","959":"Chico","960":"Redding","961":"Redding",
    "967":"Honolulu","968":"Honolulu",
    "970":"Portland","971":"Portland","972":"Portland","973":"Salem",
    "974":"Salem","975":"Medford","976":"Klamath Falls","977":"Bend",
    "978":"Eugene","979":"Eugene",
    "980":"Seattle","981":"Seattle","982":"Seattle","983":"Tacoma",
    "984":"Tacoma","985":"Olympia","986":"Bremerton","988":"Yakima",
    "989":"Wenatchee","990":"Spokane","991":"Spokane","992":"Spokane",
    "993":"Kennewick","994":"Walla Walla",
    "995":"Anchorage","996":"Fairbanks","997":"Juneau","998":"Fairbanks","999":"Nome",
}

def city_from_zip(zip_code: str) -> str:
    """Return city name from ZIP prefix table."""
    return ZIP3_CITY.get(str(zip_code).zfill(5)[:3], "")

def state_from_zip(zip_code: str) -> str:
    """Return state abbreviation from ZIP prefix table."""
    return ZIP3_STATE.get(str(zip_code).zfill(5)[:3], "")


# ──────────────────────────────────────────────────────────────
# HAVERSINE  (fixed broadcasting)
# ──────────────────────────────────────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    """Returns (N, M) distance matrix in miles."""
    R  = 3_958.8
    φ1 = np.radians(lat1)[:, None]
    φ2 = np.radians(lat2)[None, :]
    λ1 = np.radians(lon1)[:, None]
    λ2 = np.radians(lon2)[None, :]
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ──────────────────────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading ZIP code dataset…")
def build_datasets():
    zip_csv = os.path.join(DATA_DIR, "uszips.csv")

    # 1. Local cached CSV
    if os.path.exists(zip_csv):
        try:
            raw = pd.read_csv(zip_csv, dtype=str)
            df  = _normalize(raw)
            if len(df) > 100:
                return _enrich(df)
        except Exception:
            pass

    # 2. US Census ZCTA Gazetteer (public domain)
    for url in [
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip",
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_Gaz_zcta_national.zip",
    ]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if resp.content[:4] != b"PK\x03\x04":
                continue
            zf       = zipfile.ZipFile(io.BytesIO(resp.content))
            txt_name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if txt_name is None:
                continue
            with zf.open(txt_name) as fh:
                raw = pd.read_csv(fh, sep="\t", dtype=str)
            raw.columns = [c.strip().upper() for c in raw.columns]
            raw = raw.rename(columns={"GEOID":"ZIP","INTPTLAT":"Latitude","INTPTLONG":"Longitude"})
            raw["City"] = ""; raw["State"] = ""; raw["County"] = ""
            raw["Population"] = np.random.randint(1_000, 40_000, len(raw))
            df = _normalize(raw)
            df.to_csv(zip_csv, index=False)
            st.info(f"Loaded {len(df):,} ZIP codes from US Census.")
            return _enrich(df)
        except Exception:
            continue

    # 3. Synthetic fallback
    st.warning("Using built-in dataset. Place uszips.csv in data/ for full US coverage.")
    return _build_synthetic()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, fill City and State from ZIP prefix tables."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    remap = {
        "zip":"ZIP","zipcode":"ZIP","zip_code":"ZIP","geoid":"ZIP","zcta":"ZIP",
        "lat":"Latitude","latitude":"Latitude","intptlat":"Latitude",
        "lng":"Longitude","lon":"Longitude","longitude":"Longitude","intptlong":"Longitude",
        "population":"Population","pop":"Population",
        "state_id":"State","state":"State",
        "county_name":"County","county":"County",
        "city":"City",
    }
    df = df.rename(columns={k:v for k,v in remap.items() if k in df.columns})

    for col in ("Latitude","Longitude"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Latitude","Longitude"])
    df = df[df["Latitude"].between(17,72) & df["Longitude"].between(-180,-60)]

    if "ZIP"        not in df.columns: df["ZIP"]        = df.index.astype(str)
    if "Population" not in df.columns: df["Population"] = np.random.randint(1_000,40_000,len(df))
    if "City"       not in df.columns: df["City"]       = ""
    if "State"      not in df.columns: df["State"]      = ""
    if "County"     not in df.columns: df["County"]     = ""

    df["ZIP"]        = df["ZIP"].astype(str).str.strip().str.zfill(5)
    df["Population"] = pd.to_numeric(df["Population"],errors="coerce").fillna(5_000).astype(int)

    # ── Fill City from ZIP prefix table where blank / unknown ──────────
    needs_city = df["City"].isin(["","Unknown","unknown"]) | df["City"].isna()
    if needs_city.any():
        df.loc[needs_city, "City"] = (
            df.loc[needs_city, "ZIP"].apply(city_from_zip)
        )

    # ── Fill State from ZIP prefix table where blank / unknown ─────────
    needs_state = df["State"].isin(["","Unknown","unknown"]) | df["State"].isna()
    if needs_state.any():
        df.loc[needs_state, "State"] = (
            df.loc[needs_state, "ZIP"].apply(state_from_zip)
        )

    # Final fallback: anything still empty gets a generic label
    df["City"]  = df["City"].replace("", "Unknown Area").fillna("Unknown Area")
    df["State"] = df["State"].replace("", "??").fillna("??")

    return df.reset_index(drop=True)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    n = len(df)
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    east_coast   = np.clip(1-(lon+65)/20,   0,1)
    gulf_coast   = np.clip(1-(lat-25)/10,   0,1) * np.clip(1-(-lon-80)/20,0,1)
    west_coast   = np.clip(1-(-lon-115)/15, 0,1)
    inland_flood = np.clip(np.sin(np.radians(lat-30))*0.4+0.1, 0,1)
    noise = np.random.uniform(0,0.25,n)
    df["FloodRisk"]        = np.clip(east_coast*0.35+gulf_coast*0.55+inland_flood+noise,0,1)
    df["HurricaneRisk"]    = np.clip(gulf_coast*0.80+east_coast*0.35+noise*0.5,0,1)
    df["CoastalRisk"]      = np.clip(east_coast*0.50+gulf_coast*0.60+west_coast*0.45+noise*0.4,0,1)
    df["HistoricalDamage"] = np.random.randint(5_000,2_000_000,n)
    return df


def _build_synthetic() -> pd.DataFrame:
    np.random.seed(42)
    CITIES = [
        ("10001","New York","NY","New York",40.748,-73.997,8_336_817),
        ("90001","Los Angeles","CA","Los Angeles",34.052,-118.244,3_979_576),
        ("60601","Chicago","IL","Cook",41.878,-87.630,2_693_976),
        ("77001","Houston","TX","Harris",29.760,-95.370,2_304_580),
        ("85001","Phoenix","AZ","Maricopa",33.448,-112.074,1_608_139),
        ("19101","Philadelphia","PA","Philadelphia",39.953,-75.165,1_603_797),
        ("78201","San Antonio","TX","Bexar",29.424,-98.494,1_434_625),
        ("92101","San Diego","CA","San Diego",32.716,-117.161,1_386_932),
        ("75201","Dallas","TX","Dallas",32.777,-96.797,1_304_379),
        ("78701","Austin","TX","Travis",30.267,-97.743,961_855),
        ("32099","Jacksonville","FL","Duval",30.332,-81.656,949_611),
        ("94101","San Francisco","CA","San Francisco",37.775,-122.419,881_549),
        ("43201","Columbus","OH","Franklin",39.961,-82.999,905_748),
        ("28201","Charlotte","NC","Mecklenburg",35.227,-80.843,885_708),
        ("46201","Indianapolis","IN","Marion",39.768,-86.158,876_862),
        ("98101","Seattle","WA","King",47.606,-122.332,753_675),
        ("80201","Denver","CO","Denver",39.739,-104.990,727_211),
        ("37201","Nashville","TN","Davidson",36.163,-86.782,689_447),
        ("20001","Washington DC","DC","DC",38.907,-77.037,689_545),
        ("02101","Boston","MA","Suffolk",42.360,-71.059,692_600),
        ("73101","Oklahoma City","OK","Oklahoma",35.468,-97.516,681_054),
        ("89701","Las Vegas","NV","Clark",36.170,-115.140,651_319),
        ("97201","Portland","OR","Multnomah",45.505,-122.675,652_503),
        ("21201","Baltimore","MD","Baltimore",39.290,-76.612,593_490),
        ("53201","Milwaukee","WI","Milwaukee",43.039,-87.907,590_157),
        ("87101","Albuquerque","NM","Bernalillo",35.084,-106.650,560_218),
        ("85701","Tucson","AZ","Pima",32.223,-110.975,548_073),
        ("93701","Fresno","CA","Fresno",36.738,-119.787,542_107),
        ("95814","Sacramento","CA","Sacramento",38.582,-121.494,513_624),
        ("64101","Kansas City","MO","Jackson",39.100,-94.579,508_090),
        ("30301","Atlanta","GA","Fulton",33.749,-84.388,498_715),
        ("68101","Omaha","NE","Douglas",41.257,-95.935,486_051),
        ("33101","Miami","FL","Miami-Dade",25.762,-80.192,470_914),
        ("55401","Minneapolis","MN","Hennepin",44.978,-93.265,429_606),
        ("74101","Tulsa","OK","Tulsa",36.154,-95.993,413_066),
        ("27601","Raleigh","NC","Wake",35.780,-78.638,474_069),
        ("23201","Richmond","VA","Richmond",37.541,-77.436,230_436),
        ("70112","New Orleans","LA","Orleans",29.951,-90.072,383_997),
        ("77550","Galveston","TX","Galveston",29.301,-94.798,50_180),
        ("28401","Wilmington","NC","New Hanover",34.226,-77.945,123_784),
        ("29401","Charleston","SC","Charleston",32.777,-79.931,150_227),
        ("33601","Tampa","FL","Hillsborough",27.951,-82.457,399_700),
        ("33401","West Palm Beach","FL","Palm Beach",26.715,-80.053,111_955),
        ("32801","Orlando","FL","Orange",28.538,-81.379,307_573),
        ("36101","Montgomery","AL","Montgomery",32.367,-86.300,199_518),
        ("36601","Mobile","AL","Mobile",30.695,-88.040,187_041),
        ("39201","Jackson","MS","Hinds",32.299,-90.185,153_701),
        ("70801","Baton Rouge","LA","East Baton Rouge",30.452,-91.187,225_374),
        ("77002","Beaumont","TX","Jefferson",30.080,-94.127,117_796),
        ("23501","Norfolk","VA","Norfolk",36.851,-76.286,244_703),
        ("21401","Annapolis","MD","Anne Arundel",38.978,-76.492,39_474),
        ("29501","Myrtle Beach","SC","Horry",33.689,-78.887,34_695),
        ("31401","Savannah","GA","Chatham",32.084,-81.100,147_088),
        ("99501","Anchorage","AK","Anchorage",61.218,-149.900,291_247),
        ("96801","Honolulu","HI","Honolulu",21.307,-157.858,350_964),
        ("04101","Portland","ME","Cumberland",43.659,-70.257,68_408),
        ("02901","Providence","RI","Providence",41.824,-71.413,190_934),
        ("06101","Hartford","CT","Hartford",41.766,-72.685,121_054),
        ("03101","Manchester","NH","Hillsborough",42.996,-71.455,115_644),
        ("05401","Burlington","VT","Chittenden",44.476,-73.212,45_012),
        ("19801","Wilmington","DE","New Castle",39.745,-75.548,70_898),
        ("08101","Camden","NJ","Camden",39.926,-75.120,73_562),
        ("07101","Newark","NJ","Essex",40.736,-74.172,311_549),
        ("82001","Cheyenne","WY","Laramie",41.140,-104.820,65_132),
        ("58501","Bismarck","ND","Burleigh",46.808,-100.784,73_529),
        ("57501","Pierre","SD","Hughes",44.368,-100.351,14_003),
        ("59601","Helena","MT","Lewis and Clark",46.596,-112.027,32_315),
        ("83701","Boise","ID","Ada",43.615,-116.202,235_684),
        ("84101","Salt Lake City","UT","Salt Lake",40.761,-111.891,200_591),
        ("89501","Reno","NV","Washoe",39.530,-119.814,250_998),
        ("98501","Olympia","WA","Thurston",47.038,-122.905,52_555),
        ("66101","Kansas City","KS","Wyandotte",39.116,-94.627,156_607),
        ("72201","Little Rock","AR","Pulaski",34.747,-92.290,202_591),
        ("65101","Jefferson City","MO","Cole",38.577,-92.173,43_330),
        ("62701","Springfield","IL","Sangamon",39.782,-89.650,114_230),
        ("47101","New Albany","IN","Floyd",38.286,-85.824,37_841),
        ("40601","Frankfort","KY","Franklin",38.200,-84.873,28_626),
        ("37601","Kingsport","TN","Sullivan",36.549,-82.562,54_564),
        ("39401","Hattiesburg","MS","Forrest",31.327,-89.291,47_020),
        ("36201","Anniston","AL","Calhoun",33.660,-85.831,21_606),
        ("32601","Gainesville","FL","Alachua",29.652,-82.325,133_997),
        ("34201","Bradenton","FL","Manatee",27.499,-82.575,57_834),
        ("29201","Columbia","SC","Richland",34.001,-81.035,133_451),
        ("27101","Winston-Salem","NC","Forsyth",36.100,-80.244,249_545),
        ("25301","Charleston","WV","Kanawha",38.349,-81.633,46_692),
        ("17101","Harrisburg","PA","Dauphin",40.274,-76.884,50_099),
        ("14601","Rochester","NY","Monroe",43.157,-77.609,206_284),
        ("13201","Syracuse","NY","Onondaga",43.048,-76.147,142_327),
        ("12201","Albany","NY","Albany",42.653,-73.756,97_279),
        ("06901","Stamford","CT","Fairfield",41.053,-73.539,135_470),
        ("01101","Springfield","MA","Hampden",42.102,-72.590,153_984),
        ("74401","Muskogee","OK","Muskogee",35.748,-95.360,37_331),
        ("66801","Emporia","KS","Lyon",38.404,-96.181,24_916),
        ("24501","Lynchburg","VA","Lynchburg",37.414,-79.142,82_168),
        ("01801","Woburn","MA","Middlesex",42.479,-71.152,40_117),
    ]
    rows = []
    for (zipcode,city,state,county,lat,lon,pop) in CITIES:
        rows.append({"ZIP":zipcode,"City":city,"State":state,"County":county,
                     "Latitude":lat,"Longitude":lon,"Population":pop})
        for j in range(8):
            angle = j*45*np.pi/180
            d     = np.random.uniform(0.3,1.2)
            new_zip = str(int(zipcode)+j+1).zfill(5)
            rows.append({
                "ZIP":       new_zip,
                "City":      city_from_zip(new_zip) or city,
                "State":     state,
                "County":    county,
                "Latitude":  lat+d*np.sin(angle),
                "Longitude": lon+d*np.cos(angle),
                "Population":int(np.random.randint(5_000,80_000)),
            })
    return _enrich(pd.DataFrame(rows))


# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
df = build_datasets()

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.title("HydroHub Controls")
hub_count    = st.sidebar.slider("Number of Hubs",          3, 60, 15)
flood_mult   = st.sidebar.slider("Flood Weight",          0.1,3.0,1.0)
hurr_mult    = st.sidebar.slider("Hurricane Weight",      0.1,3.0,1.0)
coastal_mult = st.sidebar.slider("Coastal Storm Weight",  0.1,3.0,1.0)
st.sidebar.markdown("---")
st.sidebar.subheader("ZIP / City Lookup")
zip_lookup = st.sidebar.text_input("Enter ZIP code or city name")
st.sidebar.markdown("---")
st.sidebar.subheader("Disaster Scenario")
flood_scenario     = st.sidebar.slider("Flood Severity",     0.0,2.0,1.0)
hurricane_scenario = st.sidebar.slider("Hurricane Severity", 0.0,2.0,1.0)
top_n = st.sidebar.slider("Top N Recommended Hubs", 1, 20, 10)

# ──────────────────────────────────────────────────────────────
# WEIGHTED RISK
# ──────────────────────────────────────────────────────────────
df = df.copy()
df["RiskWeight"] = (
    df["Population"] * (
        flood_mult   * flood_scenario     * df["FloodRisk"]
      + hurr_mult    * hurricane_scenario * df["HurricaneRisk"]
      + coastal_mult *                      df["CoastalRisk"]
    ) * (1 + df["HistoricalDamage"] / 1e6)
).clip(lower=0)

# ──────────────────────────────────────────────────────────────
# HUB OPTIMISATION
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Optimising hub locations…")
def optimize_hubs(lats, lons, weights, k):
    coords = np.column_stack([lats, lons])
    w_norm = weights / (weights.sum() + 1e-9)
    idx    = np.random.choice(len(coords),
                              size=min(20_000, len(coords)*5),
                              p=w_norm, replace=True)
    km = KMeans(n_clusters=k, n_init=15, random_state=42, max_iter=500)
    km.fit(coords[idx])
    hubs = pd.DataFrame(km.cluster_centers_, columns=["Latitude","Longitude"])
    hubs["HubID"] = range(len(hubs))
    return hubs

hubs = optimize_hubs(
    df["Latitude"].values, df["Longitude"].values,
    df["RiskWeight"].values, hub_count,
)

# ──────────────────────────────────────────────────────────────
# ASSIGN NEAREST HUB
# ──────────────────────────────────────────────────────────────
dist_matrix = haversine_matrix(
    df["Latitude"].values,   df["Longitude"].values,
    hubs["Latitude"].values, hubs["Longitude"].values,
)
df["NearestHub"]    = dist_matrix.argmin(axis=1)
df["DistanceMiles"] = dist_matrix.min(axis=1)
df["TravelMinutes"] = df["DistanceMiles"] / 55.0 * 60.0 + 15.0

# ──────────────────────────────────────────────────────────────
# HUB CITY LABELS
# For each hub, find the most-populous ZIP assigned to it
# and use that ZIP's City/State as the hub's display label.
# Because City is now always populated from the ZIP table,
# this will always return a real city name.
# ──────────────────────────────────────────────────────────────
hub_city_labels = (
    df.sort_values("Population", ascending=False)
    .groupby("NearestHub")[["City","State"]]
    .first()
    .reset_index()
    .rename(columns={"NearestHub":"HubID","City":"HubCity","State":"HubState"})
)

# ──────────────────────────────────────────────────────────────
# COVERAGE TABLE
# ──────────────────────────────────────────────────────────────
coverage = (
    df.groupby("NearestHub")
    .agg(
        PopulationCovered=("Population",    "sum"),
        AvgDistanceMiles =("DistanceMiles", "mean"),
        AvgTravelMinutes =("TravelMinutes", "mean"),
        RiskExposure     =("RiskWeight",    "sum"),
        ZIPsCovered      =("ZIP",           "count"),
    )
    .reset_index()
    .rename(columns={"NearestHub":"HubID"})
    .merge(hubs,            on="HubID", how="left")
    .merge(hub_city_labels, on="HubID", how="left")
)

# ──────────────────────────────────────────────────────────────
# TOP RECOMMENDED LOCATIONS
# ──────────────────────────────────────────────────────────────
df["HubScore"] = df["RiskWeight"] / (df["TravelMinutes"] + 1)
top_recommended = (
    df.groupby(["City","State","ZIP","Latitude","Longitude"])
    .agg(TotalScore=("HubScore","sum"), PopulationCovered=("Population","sum"))
    .reset_index()
    .sort_values("TotalScore", ascending=False)
    .head(top_n)
)

# ──────────────────────────────────────────────────────────────
# HEADER METRICS
# ──────────────────────────────────────────────────────────────
st.title("HydroHub — Emergency Response Optimizer")
st.caption("Flood  |  Hurricane  |  Coastal Storm  |  Optimized hub placement")

c1,c2,c3,c4 = st.columns(4)
c1.metric("ZIPs Modeled",             f"{len(df):,}")
c2.metric("Population Modeled",       f"{df['Population'].sum():,.0f}")
c3.metric("Avg Travel Time",          f"{df['TravelMinutes'].mean():.0f} min")
c4.metric("High-Risk ZIPs (top 10%)", f"{(df['RiskWeight']>df['RiskWeight'].quantile(0.9)).sum():,}")

# ──────────────────────────────────────────────────────────────
# ZIP / CITY LOOKUP
# ──────────────────────────────────────────────────────────────
lookup_result = None
if zip_lookup.strip():
    q = zip_lookup.strip()

    # 1. Exact 5-digit ZIP
    match = df[df["ZIP"] == q.zfill(5)]
    # 2. City name contains
    if match.empty:
        match = df[df["City"].str.contains(q, case=False, na=False)]
    # 3. State abbreviation
    if match.empty:
        match = df[df["State"].str.upper() == q.upper()]

    if not match.empty:
        match         = match.sort_values("Population", ascending=False)
        lookup_result = match.iloc[0]

        city_disp  = str(lookup_result.get("City",""))
        state_disp = str(lookup_result.get("State",""))
        # If city is still blank/Unknown Area, derive from ZIP
        if not city_disp or city_disp in ("Unknown Area","Unknown","unknown"):
            city_disp = city_from_zip(lookup_result["ZIP"]) or f"ZIP {lookup_result['ZIP']}"

        st.success(f"**{city_disp}, {state_disp}** — ZIP {lookup_result['ZIP']}")

        r1c1,r1c2,r1c3,r1c4 = st.columns(4)
        r1c1.metric("City",           city_disp)
        r1c2.metric("State",          state_disp)
        r1c3.metric("ZIP Code",       lookup_result["ZIP"])
        r1c4.metric("Population",     f"{int(lookup_result['Population']):,}")

        r2c1,r2c2,r2c3,r2c4 = st.columns(4)
        r2c1.metric("Flood Risk",     f"{lookup_result['FloodRisk']:.2f}")
        r2c2.metric("Hurricane Risk", f"{lookup_result['HurricaneRisk']:.2f}")
        r2c3.metric("Coastal Risk",   f"{lookup_result['CoastalRisk']:.2f}")
        r2c4.metric("Risk Score",     f"{lookup_result['RiskWeight']:,.0f}")

        r3c1,r3c2,r3c3,r3c4 = st.columns(4)
        r3c1.metric("Nearest Hub",    f"Hub {int(lookup_result['NearestHub'])}")
        r3c2.metric("Distance",       f"{lookup_result['DistanceMiles']:.1f} mi")
        r3c3.metric("Travel Time",    f"{lookup_result['TravelMinutes']:.0f} min")
        r3c4.metric("Historical Dmg", f"${lookup_result['HistoricalDamage']:,.0f}")

        if len(match) > 1:
            with st.expander(f"See all {min(len(match),20)} matches for '{q}'"):
                show_cols = ["ZIP","City","State","Population",
                             "FloodRisk","HurricaneRisk","CoastalRisk",
                             "DistanceMiles","TravelMinutes","NearestHub"]
                st.dataframe(
                    match.head(20)[show_cols].style.format({
                        "Population":"{:,.0f}","FloodRisk":"{:.3f}",
                        "HurricaneRisk":"{:.3f}","CoastalRisk":"{:.3f}",
                        "DistanceMiles":"{:.1f}","TravelMinutes":"{:.0f}",
                    }),
                    use_container_width=True,
                )
    else:
        st.warning(f"No match for '{q}'. Try a 5-digit ZIP or city name.")

# ──────────────────────────────────────────────────────────────
# MAP
# ──────────────────────────────────────────────────────────────
clat = float(lookup_result["Latitude"])  if lookup_result is not None else 39.0
clon = float(lookup_result["Longitude"]) if lookup_result is not None else -98.0
zoom = 10 if lookup_result is not None else 4

m = folium.Map(location=[clat,clon], zoom_start=zoom, tiles="CartoDB dark_matter")

HeatMap(
    df[["Latitude","Longitude","RiskWeight"]].dropna().values.tolist(),
    radius=12, blur=18, max_zoom=10,
    gradient={0.0:"blue",0.4:"cyan",0.6:"yellow",0.8:"orange",1.0:"red"},
).add_to(m)

cluster_layer = MarkerCluster(name="ZIP / City Nodes").add_to(m)
sample   = df.sample(min(3_000,len(df)), random_state=42)
risk_max = float(df["RiskWeight"].max()) or 1.0

for _, row in sample.iterrows():
    intensity = int(min(255, row["RiskWeight"]/risk_max*255))
    color     = f"#{intensity:02x}{(255-intensity)//2:02x}00"
    city_lbl  = str(row.get("City",""))
    if not city_lbl or city_lbl in ("Unknown Area","Unknown","unknown"):
        city_lbl = city_from_zip(row["ZIP"]) or row["ZIP"]
    folium.CircleMarker(
        location=[row["Latitude"],row["Longitude"]],
        radius=3, color=color, fill=True, fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>{city_lbl}, {row.get('State','')}</b><br>"
            f"ZIP: {row['ZIP']}<br>"
            f"Pop: {int(row['Population']):,}<br>"
            f"Flood: {row['FloodRisk']:.2f} | "
            f"Hurr: {row['HurricaneRisk']:.2f} | "
            f"Coast: {row['CoastalRisk']:.2f}<br>"
            f"Hub {int(row['NearestHub'])} — {row['TravelMinutes']:.0f} min",
            max_width=220,
        ),
    ).add_to(cluster_layer)

for _, hub in hubs.iterrows():
    cov_row  = coverage[coverage["HubID"]==hub["HubID"]]
    pop_cov  = int(cov_row["PopulationCovered"].iloc[0])  if len(cov_row) else 0
    avg_min  = float(cov_row["AvgTravelMinutes"].iloc[0]) if len(cov_row) else 0.0
    zips_cov = int(cov_row["ZIPsCovered"].iloc[0])        if len(cov_row) else 0
    hub_city = str(cov_row["HubCity"].iloc[0])            if len(cov_row) else ""
    hub_st   = str(cov_row["HubState"].iloc[0])           if len(cov_row) else ""
    if not hub_city or hub_city in ("Unknown Area","Unknown","unknown"):
        hub_city = city_from_zip(str(hub.get("ZIP",""))) or "Hub Area"
    folium.Marker(
        location=[hub["Latitude"],hub["Longitude"]],
        icon=folium.Icon(color="green",icon="star",prefix="fa"),
        tooltip=f"Hub {int(hub['HubID'])} — {hub_city}, {hub_st}",
        popup=folium.Popup(
            f"<b>Hub {int(hub['HubID'])}</b><br>"
            f"Near: <b>{hub_city}, {hub_st}</b><br>"
            f"Pop Covered: {pop_cov:,}<br>"
            f"ZIPs Covered: {zips_cov}<br>"
            f"Avg Travel: {avg_min:.0f} min",
            max_width=200,
        ),
    ).add_to(m)
    folium.Circle(
        location=[hub["Latitude"],hub["Longitude"]],
        radius=322_000, color="cyan", weight=1, fill=True, fill_opacity=0.03,
    ).add_to(m)

for _, row in top_recommended.iterrows():
    city_lbl = str(row.get("City",""))
    if not city_lbl or city_lbl in ("Unknown Area","Unknown","unknown"):
        city_lbl = city_from_zip(row["ZIP"]) or row["ZIP"]
    folium.Marker(
        location=[row["Latitude"],row["Longitude"]],
        icon=folium.Icon(color="purple",icon="bolt",prefix="fa"),
        tooltip=f"Recommended — {city_lbl}, {row.get('State','')}",
        popup=folium.Popup(
            f"<b>Recommended Hub</b><br>"
            f"{city_lbl}, {row.get('State','')} ({row['ZIP']})<br>"
            f"Score: {row['TotalScore']:,.0f}<br>"
            f"Pop: {int(row['PopulationCovered']):,}",
            max_width=200,
        ),
    ).add_to(m)

if lookup_result is not None:
    city_lbl  = str(lookup_result.get("City",""))
    state_lbl = str(lookup_result.get("State",""))
    if not city_lbl or city_lbl in ("Unknown Area","Unknown","unknown"):
        city_lbl = city_from_zip(lookup_result["ZIP"]) or lookup_result["ZIP"]
    folium.Marker(
        location=[float(lookup_result["Latitude"]),float(lookup_result["Longitude"])],
        icon=folium.Icon(color="red",icon="crosshairs",prefix="fa"),
        popup=f"{city_lbl}, {state_lbl}",
        tooltip=f"{city_lbl}, {state_lbl}",
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=None, height=650, returned_objects=[])

# ──────────────────────────────────────────────────────────────
# TABS  — all CSVs include City and State
# ──────────────────────────────────────────────────────────────
st.divider()
tab1,tab2,tab3,tab4 = st.tabs(
    ["Hub Coverage","High-Risk Areas","Hub Locations","Recommended Hubs"]
)

with tab1:
    st.subheader("Hub Coverage — City, State & Statistics")
    disp = coverage[["HubID","HubCity","HubState",
                      "PopulationCovered","ZIPsCovered",
                      "AvgDistanceMiles","AvgTravelMinutes","RiskExposure"]] \
                   .rename(columns={"HubCity":"City","HubState":"State"}) \
                   .sort_values("RiskExposure", ascending=False)
    st.dataframe(
        disp.style.format({
            "PopulationCovered":"{:,.0f}","AvgDistanceMiles":"{:.1f}",
            "AvgTravelMinutes":"{:.0f}","RiskExposure":"{:,.0f}",
        }),
        use_container_width=True,
    )
    st.download_button("Download CSV", disp.to_csv(index=False),
                       "hub_coverage.csv","text/csv")

with tab2:
    st.subheader("Top 100 Highest-Risk ZIP Codes")
    cols = ["ZIP","City","State","Population","RiskWeight",
            "FloodRisk","HurricaneRisk","CoastalRisk",
            "NearestHub","DistanceMiles","TravelMinutes"]
    top100 = df.sort_values("RiskWeight",ascending=False).head(100)[cols]
    st.dataframe(
        top100.style.format({
            "Population":"{:,.0f}","RiskWeight":"{:,.0f}",
            "FloodRisk":"{:.3f}","HurricaneRisk":"{:.3f}","CoastalRisk":"{:.3f}",
            "DistanceMiles":"{:.1f}","TravelMinutes":"{:.0f}",
        }).background_gradient(subset=["RiskWeight"],cmap="Reds"),
        use_container_width=True,
    )
    st.download_button("Download CSV", top100.to_csv(index=False),
                       "high_risk_zips.csv","text/csv")

with tab3:
    st.subheader("Optimised Hub Locations — City & State")
    hubs_disp = hubs.merge(hub_city_labels,on="HubID",how="left") \
                    .rename(columns={"HubCity":"City","HubState":"State"})
    st.dataframe(
        hubs_disp[["HubID","City","State","Latitude","Longitude"]],
        use_container_width=True,
    )
    st.download_button("Download CSV",
                       hubs_disp[["HubID","City","State","Latitude","Longitude"]] \
                           .to_csv(index=False),
                       "hubs.csv","text/csv")

with tab4:
    st.subheader(f"Top {top_n} Recommended Hub Locations")
    rec_cols = ["City","State","ZIP","Latitude","Longitude",
                "TotalScore","PopulationCovered"]
    st.dataframe(
        top_recommended[rec_cols].style.format({
            "TotalScore":"{:,.0f}","PopulationCovered":"{:,.0f}",
        }),
        use_container_width=True,
    )
    st.download_button("Download CSV",
                       top_recommended[rec_cols].to_csv(index=False),
                       "recommended_hubs.csv","text/csv")
