from typing import Dict, List
from itertools import chain
from data_cache import pandas_cache
import censusdata
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler
import enum
from operator import attrgetter
from IPython.display import display, HTML


class Nameable(str):
    @classmethod
    @property
    def names(cls) -> List[str]:
        return list(map(attrgetter("name"), cls))


class LanguageVars(Nameable, enum.Enum):
    C16001_001E = "total speakers"
    C16001_005E = "spanish speakers"
    C16001_020E = "korean speakers"
    C16001_026E = "vietnamese speakers"
    C16001_035E = "arabic speakers"
    C16001_023E = "chinese (incl. mandarin, cantonese) speakers"
    C16001_008E = "french, haitian, or cajun speakers"
    C16001_011E = "german or other west germanic languages speakers"
    C16001_014E = "russian, polish, or other slavic languages speakers"
    C16001_017E = "other indo-european languages speakers"
    C16001_029E = "tagalog (incl. filipino) speakers"
    C16001_032E = "other asian and pacific island languages speakers"
    C16001_038E = "other and unspecified languages speakers"


class DetailedLanguageVars(Nameable, enum.Enum):
    B16001_001E = "total speakers"
    B16001_005E = "spanish speakers"
    B16001_008E = "french (incl. cajun) speakers"
    B16001_011E = "haitian speakers"
    B16001_014E = "italian speakers"
    B16001_017E = "portuguese speakers"
    B16001_020E = "german speakers"
    B16001_023E = (
        "yiddish, pennsylvania dutch or other west germanic languages speakers"
    )
    B16001_026E = "greek speakers"
    B16001_029E = "russian speakers"
    B16001_032E = "polish speakers"
    B16001_035E = "serbo-croatian speakers"
    B16001_038E = "ukrainian or other slavic languages speakers"
    B16001_041E = "armenian speakers"
    B16001_044E = "persian (incl. farsi, dari) speakers"
    B16001_047E = "gujarati speakers"
    B16001_050E = "hindi speakers"
    B16001_053E = "urdu speakers"
    B16001_056E = "punjabi speakers"
    B16001_059E = "bengali speakers"
    B16001_062E = "nepali, marathi, or other indic languages speakers"
    B16001_065E = "other indo-european languages speakers"
    B16001_068E = "telugu speakers"
    B16001_071E = "tamil speakers"
    B16001_074E = "malayalam, kannada, or other dravidian languages speakers"
    B16001_077E = "chinese (incl. mandarin, cantonese) speakers"
    B16001_080E = "japanese speakers"
    B16001_083E = "korean speakers"
    B16001_086E = "hmong speakers"
    B16001_089E = "vietnamese speakers"
    B16001_092E = "khmer speakers"
    B16001_095E = "thai, lao, or other tai-kadai languages speakers"
    B16001_098E = "other languages of asia speakers"
    B16001_101E = "tagalog (incl. filipino) speakers"
    B16001_104E = "ilocano, samoan, hawaiian, or other austronesian languages speakers"
    B16001_107E = "arabic speakers"
    B16001_110E = "hebrew speakers"
    B16001_113E = "amharic, somali, or other afro-asiatic languages speakers"
    B16001_116E = "yoruba, twi, igbo, or other languages of western africa speakers"
    B16001_119E = (
        "swahili or other languages of central, eastern, and southern africa speakers"
    )
    B16001_122E = "navajo speakers"
    B16001_125E = "other native languages of north america speakers"
    B16001_128E = "other and unspecified languages speakers"


class PublicAssistanceVars(Nameable, enum.Enum):
    B19058_001E = "total public assistance population"
    B19058_002E = "received public assistance"


class PovertyLevelVars(Nameable, enum.Enum):
    B17026_001E = "total poverty"
    B17026_002E = "under 0.5"
    B17026_003E = "0.5 to 0.74"
    B17026_004E = "0.75 to 0.99"
    B17026_005E = "1.00 to 1.24"
    B17026_006E = "1.25 to 1.49"
    B17026_007E = "1.50 to 1.74"
    B17026_008E = "1.75 to 1.84"


class TotalPopulationVars(Nameable, enum.Enum):
    B01003_001E = "total population"


@pandas_cache
def _get_frame_for_all_county_vars() -> pd.DataFrame:
    """Just get everything we want all at once

    Originally I wrote this code to download each enum of variables separately,
    but each call to censusdata.download takes several seconds, so that was turning into
    several minutes. It's nice to be able to pretend I have separate frames for each enum,
    so the interface to get_frame_for_var keeps that the same, but getting all the data at once
    makes life a lot faster.
    """
    all_vars = list(
        chain(
            LanguageVars.names,
            PublicAssistanceVars.names,
            PovertyLevelVars.names,
            TotalPopulationVars.names,
        )
    )

    state_data = censusdata.download(
        "acs5",
        2019,
        censusdata.censusgeo([("state", "*")]),
        all_vars,
    )

    county_data = censusdata.download(
        "acs5",
        2019,
        censusdata.censusgeo([("county", "*")]),
        all_vars,
    )

    return pd.concat([state_data, county_data])


def get_frame_for_county_vars(CensusVars: enum.Enum) -> pd.DataFrame:
    """Return a subset of all the data we want with nicely named columns and a multi-index"""
    data = _get_frame_for_all_county_vars()[[val.name for val in CensusVars]].copy()
    data.columns = [val.value for val in CensusVars]
    data["state fips"] = data.index.map(lambda idx: idx.params()[0][-1])
    data["place name"] = data.index.map(lambda idx: idx.name)
    sort_col = [val.value for val in CensusVars][0]

    data = (
        data.set_index(["state fips", "place name"])
        .sort_values(sort_col, ascending=False)
        .sort_index(level=0, sort_remaining=False)
    )
    return data


@pandas_cache
def get_frame_for_state_vars(CensusVars: enum.Enum) -> pd.DataFrame:
    data = censusdata.download(
        "acs5", 2019, censusdata.censusgeo([("state", "*")]), CensusVars.names
    )
    data.columns = [val.value for val in CensusVars]
    data["state fips"] = data.index.map(lambda idx: idx.params()[0][-1])
    data["place name"] = data.index.map(lambda idx: idx.name)
    data.set_index(["state fips", "place name"], inplace=True)
    return data


@pandas_cache
def get_frame_for_tribal_areas(CensusVars: enum.Enum) -> pd.DataFrame:
    data = censusdata.download(
        "acs5",
        2019,
        censusdata.censusgeo(
            [("american indian area/alaska native area/hawaiian home land", "*")]
        ),
        CensusVars.names,
    )
    data.columns = [val.value for val in CensusVars]
    data["place name"] = data.index.map(lambda n: n.name)
    data = data.set_index("place name").sort_index()
    return data


def get_state_fips_codes() -> Dict[str, str]:
    """The census talks about everything in terms of the FIPS codes, so it's nice to have a dict of them"""
    # These are generated by the following code:
    # return {
    #     name: geo.params()[0][-1]
    #     for name, geo in censusdata.geographies(
    #         censusdata.censusgeo([("state", "*")]), "acs5", 2019
    #     ).items()
    # }

    return {
        "Alabama": "01",
        "Alaska": "02",
        "Arizona": "04",
        "Arkansas": "05",
        "California": "06",
        "Colorado": "08",
        "Delaware": "10",
        "District of Columbia": "11",
        "Connecticut": "09",
        "Florida": "12",
        "Georgia": "13",
        "Idaho": "16",
        "Hawaii": "15",
        "Illinois": "17",
        "Indiana": "18",
        "Iowa": "19",
        "Kansas": "20",
        "Kentucky": "21",
        "Louisiana": "22",
        "Maine": "23",
        "Maryland": "24",
        "Massachusetts": "25",
        "Michigan": "26",
        "Minnesota": "27",
        "Mississippi": "28",
        "Missouri": "29",
        "Montana": "30",
        "Nebraska": "31",
        "Nevada": "32",
        "New Hampshire": "33",
        "New Jersey": "34",
        "New Mexico": "35",
        "New York": "36",
        "North Carolina": "37",
        "North Dakota": "38",
        "Ohio": "39",
        "Oklahoma": "40",
        "Oregon": "41",
        "Pennsylvania": "42",
        "Rhode Island": "44",
        "South Carolina": "45",
        "South Dakota": "46",
        "Tennessee": "47",
        "Texas": "48",
        "Vermont": "50",
        "Utah": "49",
        "Virginia": "51",
        "Washington": "53",
        "West Virginia": "54",
        "Wisconsin": "55",
        "Wyoming": "56",
        "Puerto Rico": "72",
    }


def get_tribal_area_names() -> Dict[str, str]:
    return {
        "Acoma Pueblo and Off-Reservation Trust Land, NM": "Acoma Pueblo and Off-Reservation Trust Land, NM",
        "Adais Caddo (state) SDTSA, LA": "Adais Caddo (state) SDTSA, LA",
        "Agua Caliente Indian Reservation and Off-Reservation Trust Land, CA": "Agua Caliente Indian Reservation and Off-Reservation Trust Land, CA",
        "Akhiok ANVSA, AK": "Akhiok ANVSA, AK",
        "Akiachak ANVSA, AK": "Akiachak ANVSA, AK",
        "Akiak ANVSA, AK": "Akiak ANVSA, AK",
        "Akutan ANVSA, AK": "Akutan ANVSA, AK",
        "Alabama-Coushatta Reservation and Off-Reservation Trust Land, TX": "Alabama-Coushatta Reservation and Off-Reservation Trust Land, TX",
        "Alakanuk ANVSA, AK": "Alakanuk ANVSA, AK",
        "Alatna ANVSA, AK": "Alatna ANVSA, AK",
        "Aleknagik ANVSA, AK": "Aleknagik ANVSA, AK",
        "Algaaciq ANVSA, AK": "Algaaciq ANVSA, AK",
        "Allakaket ANVSA, AK": "Allakaket ANVSA, AK",
        "Allegany Reservation, NY": "Allegany Reservation, NY",
        "Alturas Indian Rancheria, CA": "Alturas Indian Rancheria, CA",
        "Ambler ANVSA, AK": "Ambler ANVSA, AK",
        "Anahola (Agricultural) Hawaiian Home Land, HI": "Anahola (Agricultural) Hawaiian Home Land, HI",
        "Anahola (Residential) Hawaiian Home Land, HI": "Anahola (Residential) Hawaiian Home Land, HI",
        "Anaktuvuk Pass ANVSA, AK": "Anaktuvuk Pass ANVSA, AK",
        "Andreafsky ANVSA, AK": "Andreafsky ANVSA, AK",
        "Angoon ANVSA, AK": "Angoon ANVSA, AK",
        "Aniak ANVSA, AK": "Aniak ANVSA, AK",
        "Annette Island Reserve, AK": "Annette Island Reserve, AK",
        "Anvik ANVSA, AK": "Anvik ANVSA, AK",
        "Apache Choctaw (state) SDTSA, LA": "Apache Choctaw (state) SDTSA, LA",
        "Arctic Village ANVSA, AK": "Arctic Village ANVSA, AK",
        "Aroostook Band of Micmac Trust Land, ME": "Aroostook Band of Micmac Trust Land, ME",
        "Atka ANVSA, AK": "Atka ANVSA, AK",
        "Atmautluak ANVSA, AK": "Atmautluak ANVSA, AK",
        "Atqasuk ANVSA, AK": "Atqasuk ANVSA, AK",
        "Auburn Rancheria and Off-Reservation Trust Land, CA": "Auburn Rancheria and Off-Reservation Trust Land, CA",
        "Augustine Reservation, CA": "Augustine Reservation, CA",
        "Bad River Reservation, WI": "Bad River Reservation, WI",
        "Barona Reservation and Off-Reservation Trust Land, CA": "Barona Reservation and Off-Reservation Trust Land, CA",
        "Barrow ANVSA, AK": "Barrow ANVSA, AK",
        "Battle Mountain Reservation and Off-Reservation Trust Land, NV": "Battle Mountain Reservation and Off-Reservation Trust Land, NV",
        "Bay Mills Reservation and Off-Reservation Trust Land, MI": "Bay Mills Reservation and Off-Reservation Trust Land, MI",
        "Beaver ANVSA, AK": "Beaver ANVSA, AK",
        "Beaver Creek SDTSA, SC": "Beaver Creek SDTSA, SC",
        "Belkofski ANVSA, AK": "Belkofski ANVSA, AK",
        "Benton Paiute Reservation and Off-Reservation Trust Land, CA": "Benton Paiute Reservation and Off-Reservation Trust Land, CA",
        "Berry Creek Rancheria and Off-Reservation Trust Land, CA": "Berry Creek Rancheria and Off-Reservation Trust Land, CA",
        "Bethel ANVSA, AK": "Bethel ANVSA, AK",
        "Big Bend Rancheria, CA": "Big Bend Rancheria, CA",
        "Big Cypress Reservation, FL": "Big Cypress Reservation, FL",
        "Big Lagoon Rancheria, CA": "Big Lagoon Rancheria, CA",
        "Big Pine Reservation and Off-Reservation Trust Land, CA": "Big Pine Reservation and Off-Reservation Trust Land, CA",
        "Big Sandy Rancheria and Off-Reservation Trust Land, CA": "Big Sandy Rancheria and Off-Reservation Trust Land, CA",
        "Big Valley Rancheria, CA": "Big Valley Rancheria, CA",
        "Bill Moore's ANVSA, AK": "Bill Moore's ANVSA, AK",
        "Birch Creek ANVSA, AK": "Birch Creek ANVSA, AK",
        "Bishop Reservation, CA": "Bishop Reservation, CA",
        "Blackfeet Indian Reservation and Off-Reservation Trust Land, MT": "Blackfeet Indian Reservation and Off-Reservation Trust Land, MT",
        "Blue Lake Rancheria and Off-Reservation Trust Land, CA": "Blue Lake Rancheria and Off-Reservation Trust Land, CA",
        "Bois Forte Reservation and Off-Reservation Trust Land, MN": "Bois Forte Reservation and Off-Reservation Trust Land, MN",
        "Brevig Mission ANVSA, AK": "Brevig Mission ANVSA, AK",
        "Bridgeport Reservation and Off-Reservation Trust Land, CA": "Bridgeport Reservation and Off-Reservation Trust Land, CA",
        "Brighton Reservation, FL": "Brighton Reservation, FL",
        "Buckland ANVSA, AK": "Buckland ANVSA, AK",
        "Burns Paiute Indian Colony and Off-Reservation Trust Land, OR": "Burns Paiute Indian Colony and Off-Reservation Trust Land, OR",
        "Cabazon Reservation, CA": "Cabazon Reservation, CA",
        "Caddo-Wichita-Delaware OTSA, OK": "Caddo-Wichita-Delaware OTSA, OK",
        "Cahuilla Reservation, CA": "Cahuilla Reservation, CA",
        "Campbell Ranch, NV": "Campbell Ranch, NV",
        "Campo Indian Reservation, CA": "Campo Indian Reservation, CA",
        "Cantwell ANVSA, AK": "Cantwell ANVSA, AK",
        "Canyon Village ANVSA, AK": "Canyon Village ANVSA, AK",
        "Capitan Grande Reservation, CA": "Capitan Grande Reservation, CA",
        "Carson Colony, NV": "Carson Colony, NV",
        "Catawba Reservation, SC": "Catawba Reservation, SC",
        "Cattaraugus Reservation, NY": "Cattaraugus Reservation, NY",
        "Cayuga Nation TDSA, NY": "Cayuga Nation TDSA, NY",
        "Cedarville Rancheria and Off-Reservation Trust Land, CA": "Cedarville Rancheria and Off-Reservation Trust Land, CA",
        "Celilo Village Trust Land, OR": "Celilo Village Trust Land, OR",
        "Chalkyitsik ANVSA, AK": "Chalkyitsik ANVSA, AK",
        "Chefornak ANVSA, AK": "Chefornak ANVSA, AK",
        "Chehalis Reservation and Off-Reservation Trust Land, WA": "Chehalis Reservation and Off-Reservation Trust Land, WA",
        "Chemehuevi Reservation, CA": "Chemehuevi Reservation, CA",
        "Chenega ANVSA, AK": "Chenega ANVSA, AK",
        "Cher-O-Creek SDTSA, AL": "Cher-O-Creek SDTSA, AL",
        "Cherokee OTSA, OK": "Cherokee OTSA, OK",
        "Cherokee Tribe of Northeast Alabama (state) SDTSA, AL": "Cherokee Tribe of Northeast Alabama (state) SDTSA, AL",
        "Chevak ANVSA, AK": "Chevak ANVSA, AK",
        "Cheyenne River Reservation and Off-Reservation Trust Land, SD": "Cheyenne River Reservation and Off-Reservation Trust Land, SD",
        "Cheyenne-Arapaho OTSA, OK": "Cheyenne-Arapaho OTSA, OK",
        "Chickahominy (state) SDTSA, VA": "Chickahominy (state) SDTSA, VA",
        "Chickaloon ANVSA, AK": "Chickaloon ANVSA, AK",
        "Chickasaw OTSA, OK": "Chickasaw OTSA, OK",
        "Chicken Ranch Rancheria and Off-Reservation Trust Land, CA": "Chicken Ranch Rancheria and Off-Reservation Trust Land, CA",
        "Chignik ANVSA, AK": "Chignik ANVSA, AK",
        "Chignik Lagoon ANVSA, AK": "Chignik Lagoon ANVSA, AK",
        "Chignik Lake ANVSA, AK": "Chignik Lake ANVSA, AK",
        "Chilkat ANVSA, AK": "Chilkat ANVSA, AK",
        "Chilkoot ANVSA, AK": "Chilkoot ANVSA, AK",
        "Chistochina ANVSA, AK": "Chistochina ANVSA, AK",
        "Chitimacha Reservation, LA": "Chitimacha Reservation, LA",
        "Chitina ANVSA, AK": "Chitina ANVSA, AK",
        "Choctaw OTSA, OK": "Choctaw OTSA, OK",
        "Chuathbaluk ANVSA, AK": "Chuathbaluk ANVSA, AK",
        "Chuloonawick ANVSA, AK": "Chuloonawick ANVSA, AK",
        "Circle ANVSA, AK": "Circle ANVSA, AK",
        "Citizen Potawatomi Nation-Absentee Shawnee OTSA, OK": "Citizen Potawatomi Nation-Absentee Shawnee OTSA, OK",
        "Clarks Point ANVSA, AK": "Clarks Point ANVSA, AK",
        "Clifton Choctaw (state) SDTSA, LA": "Clifton Choctaw (state) SDTSA, LA",
        "Coconut Creek Trust Land, FL": "Coconut Creek Trust Land, FL",
        "Cocopah Reservation, AZ": "Cocopah Reservation, AZ",
        "Coeur d'Alene Reservation, ID": "Coeur d'Alene Reservation, ID",
        "Coharie (state) SDTSA, NC": "Coharie (state) SDTSA, NC",
        "Cold Springs Rancheria, CA": "Cold Springs Rancheria, CA",
        "Colorado River Indian Reservation, AZ--CA": "Colorado River Indian Reservation, AZ--CA",
        "Colusa Rancheria, CA": "Colusa Rancheria, CA",
        "Colville Reservation and Off-Reservation Trust Land, WA": "Colville Reservation and Off-Reservation Trust Land, WA",
        "Coos, Lower Umpqua, and Siuslaw Reservation and Off-Reservation Trust Land, OR": "Coos, Lower Umpqua, and Siuslaw Reservation and Off-Reservation Trust Land, OR",
        "Copper Center ANVSA, AK": "Copper Center ANVSA, AK",
        "Coquille Reservation, OR": "Coquille Reservation, OR",
        "Cortina Indian Rancheria, CA": "Cortina Indian Rancheria, CA",
        "Council ANVSA, AK": "Council ANVSA, AK",
        "Coushatta Reservation and Off-Reservation Trust Land, LA": "Coushatta Reservation and Off-Reservation Trust Land, LA",
        "Cow Creek Reservation and Off-Reservation Trust Land, OR": "Cow Creek Reservation and Off-Reservation Trust Land, OR",
        "Cowlitz Reservation, WA": "Cowlitz Reservation, WA",
        "Coyote Valley Reservation, CA": "Coyote Valley Reservation, CA",
        "Craig ANVSA, AK": "Craig ANVSA, AK",
        "Creek OTSA, OK": "Creek OTSA, OK",
        "Creek/Seminole joint-use OTSA, OK": "Creek/Seminole joint-use OTSA, OK",
        "Crooked Creek ANVSA, AK": "Crooked Creek ANVSA, AK",
        "Crow Creek Reservation, SD": "Crow Creek Reservation, SD",
        "Crow Reservation and Off-Reservation Trust Land, MT": "Crow Reservation and Off-Reservation Trust Land, MT",
        "Deering ANVSA, AK": "Deering ANVSA, AK",
        "Dillingham ANVSA, AK": "Dillingham ANVSA, AK",
        "Dot Lake ANVSA, AK": "Dot Lake ANVSA, AK",
        "Douglas ANVSA, AK": "Douglas ANVSA, AK",
        "Dresslerville Colony, NV": "Dresslerville Colony, NV",
        "Dry Creek Rancheria and Off-Reservation Trust Land, CA": "Dry Creek Rancheria and Off-Reservation Trust Land, CA",
        "Duck Valley Reservation, NV--ID": "Duck Valley Reservation, NV--ID",
        "Duckwater Reservation, NV": "Duckwater Reservation, NV",
        "Eagle ANVSA, AK": "Eagle ANVSA, AK",
        "East Kapolei Hawaiian Home Land, HI": "East Kapolei Hawaiian Home Land, HI",
        "Eastern Cherokee Reservation, NC": "Eastern Cherokee Reservation, NC",
        "Eastern Chickahominy (state) SDTSA, VA": "Eastern Chickahominy (state) SDTSA, VA",
        "Eastern Shawnee OTSA, OK": "Eastern Shawnee OTSA, OK",
        "Echota Cherokee (state) SDTSA, AL": "Echota Cherokee (state) SDTSA, AL",
        "Eek ANVSA, AK": "Eek ANVSA, AK",
        "Egegik ANVSA, AK": "Egegik ANVSA, AK",
        "Eklutna ANVSA, AK": "Eklutna ANVSA, AK",
        "Ekuk ANVSA, AK": "Ekuk ANVSA, AK",
        "Ekwok ANVSA, AK": "Ekwok ANVSA, AK",
        "Elim ANVSA, AK": "Elim ANVSA, AK",
        "Elk Valley Rancheria and Off-Reservation Trust Land, CA": "Elk Valley Rancheria and Off-Reservation Trust Land, CA",
        "Elko Colony, NV": "Elko Colony, NV",
        "Ely Reservation, NV": "Ely Reservation, NV",
        "Emmonak ANVSA, AK": "Emmonak ANVSA, AK",
        "Enterprise Rancheria and Off-Reservation Trust Land, CA": "Enterprise Rancheria and Off-Reservation Trust Land, CA",
        "Evansville ANVSA, AK": "Evansville ANVSA, AK",
        "Ewiiaapaayp Reservation, CA": "Ewiiaapaayp Reservation, CA",
        "Eyak ANVSA, AK": "Eyak ANVSA, AK",
        "Fallon Paiute-Shoshone Colony and Off-Reservation Trust Land, NV": "Fallon Paiute-Shoshone Colony and Off-Reservation Trust Land, NV",
        "Fallon Paiute-Shoshone Reservation and Off-Reservation Trust Land, NV": "Fallon Paiute-Shoshone Reservation and Off-Reservation Trust Land, NV",
        "False Pass ANVSA, AK": "False Pass ANVSA, AK",
        "Flandreau Reservation, SD": "Flandreau Reservation, SD",
        "Flathead Reservation, MT": "Flathead Reservation, MT",
        "Fond du Lac Reservation and Off-Reservation Trust Land, MN--WI": "Fond du Lac Reservation and Off-Reservation Trust Land, MN--WI",
        "Forest County Potawatomi Community and Off-Reservation Trust Land, WI": "Forest County Potawatomi Community and Off-Reservation Trust Land, WI",
        "Fort Apache Reservation, AZ": "Fort Apache Reservation, AZ",
        "Fort Belknap Reservation and Off-Reservation Trust Land, MT": "Fort Belknap Reservation and Off-Reservation Trust Land, MT",
        "Fort Berthold Reservation, ND": "Fort Berthold Reservation, ND",
        "Fort Bidwell Reservation and Off-Reservation Trust Land, CA": "Fort Bidwell Reservation and Off-Reservation Trust Land, CA",
        "Fort Hall Reservation and Off-Reservation Trust Land, ID": "Fort Hall Reservation and Off-Reservation Trust Land, ID",
        "Fort Independence Reservation, CA": "Fort Independence Reservation, CA",
        "Fort McDermitt Indian Reservation, NV--OR": "Fort McDermitt Indian Reservation, NV--OR",
        "Fort McDowell Yavapai Nation Reservation, AZ": "Fort McDowell Yavapai Nation Reservation, AZ",
        "Fort Mojave Reservation and Off-Reservation Trust Land, AZ--CA--NV": "Fort Mojave Reservation and Off-Reservation Trust Land, AZ--CA--NV",
        "Fort Peck Indian Reservation and Off-Reservation Trust Land, MT": "Fort Peck Indian Reservation and Off-Reservation Trust Land, MT",
        "Fort Pierce Reservation, FL": "Fort Pierce Reservation, FL",
        "Fort Sill Apache Indian Reservation, NM": "Fort Sill Apache Indian Reservation, NM",
        "Fort Yukon ANVSA, AK": "Fort Yukon ANVSA, AK",
        "Fort Yuma Indian Reservation, CA--AZ": "Fort Yuma Indian Reservation, CA--AZ",
        "Four Winds Cherokee (state) SDTSA, LA": "Four Winds Cherokee (state) SDTSA, LA",
        "Gakona ANVSA, AK": "Gakona ANVSA, AK",
        "Galena ANVSA, AK": "Galena ANVSA, AK",
        "Gambell ANVSA, AK": "Gambell ANVSA, AK",
        "Georgetown ANVSA, AK": "Georgetown ANVSA, AK",
        "Gila River Indian Reservation, AZ": "Gila River Indian Reservation, AZ",
        "Golden Hill Paugussett (state) Reservation, CT": "Golden Hill Paugussett (state) Reservation, CT",
        "Golovin ANVSA, AK": "Golovin ANVSA, AK",
        "Goodnews Bay ANVSA, AK": "Goodnews Bay ANVSA, AK",
        "Goshute Reservation, NV--UT": "Goshute Reservation, NV--UT",
        "Grand Portage Reservation and Off-Reservation Trust Land, MN": "Grand Portage Reservation and Off-Reservation Trust Land, MN",
        "Grand Ronde Community, OR": "Grand Ronde Community, OR",
        "Grand Traverse Reservation and Off-Reservation Trust Land, MI": "Grand Traverse Reservation and Off-Reservation Trust Land, MI",
        "Grayling ANVSA, AK": "Grayling ANVSA, AK",
        "Greenville Rancheria, CA": "Greenville Rancheria, CA",
        "Grindstone Indian Rancheria, CA": "Grindstone Indian Rancheria, CA",
        "Guidiville Rancheria and Off-Reservation Trust Land, CA": "Guidiville Rancheria and Off-Reservation Trust Land, CA",
        "Gulkana ANVSA, AK": "Gulkana ANVSA, AK",
        "Haiku Hawaiian Home Land, HI": "Haiku Hawaiian Home Land, HI",
        "Haliwa-Saponi (state) SDTSA, NC": "Haliwa-Saponi (state) SDTSA, NC",
        "Hamilton ANVSA, AK": "Hamilton ANVSA, AK",
        "Hanapepe Hawaiian Home Land, HI": "Hanapepe Hawaiian Home Land, HI",
        "Hannahville Indian Community and Off-Reservation Trust Land, MI": "Hannahville Indian Community and Off-Reservation Trust Land, MI",
        "Hassanamisco Reservation (state), MA": "Hassanamisco Reservation (state), MA",
        "Havasupai Reservation, AZ": "Havasupai Reservation, AZ",
        "Healy Lake ANVSA, AK": "Healy Lake ANVSA, AK",
        "Ho-Chunk Nation Reservation and Off-Reservation Trust Land, WI--MN": "Ho-Chunk Nation Reservation and Off-Reservation Trust Land, WI--MN",
        "Hoh Indian Reservation and Off-Reservation Trust Land, WA": "Hoh Indian Reservation and Off-Reservation Trust Land, WA",
        "Hollywood Reservation, FL": "Hollywood Reservation, FL",
        "Holy Cross ANVSA, AK": "Holy Cross ANVSA, AK",
        "Homuula-Upper Piihonua Hawaiian Home Land, HI": "Homuula-Upper Piihonua Hawaiian Home Land, HI",
        "Honokaia Hawaiian Home Land, HI": "Honokaia Hawaiian Home Land, HI",
        "Honokowai Hawaiian Home Land, HI": "Honokowai Hawaiian Home Land, HI",
        "Honolulu Makai Hawaiian Home Land, HI": "Honolulu Makai Hawaiian Home Land, HI",
        "Honomu Hawaiian Home Land, HI": "Honomu Hawaiian Home Land, HI",
        "Hoolehua-Palaaau Hawaiian Home Land, HI": "Hoolehua-Palaaau Hawaiian Home Land, HI",
        "Hoonah ANVSA, AK": "Hoonah ANVSA, AK",
        "Hoopa Valley Reservation, CA": "Hoopa Valley Reservation, CA",
        "Hooper Bay ANVSA, AK": "Hooper Bay ANVSA, AK",
        "Hopi Reservation and Off-Reservation Trust Land, AZ": "Hopi Reservation and Off-Reservation Trust Land, AZ",
        "Hopland Rancheria, CA": "Hopland Rancheria, CA",
        "Houlton Maliseet Reservation and Off-Reservation Trust Land, ME": "Houlton Maliseet Reservation and Off-Reservation Trust Land, ME",
        "Hualapai Indian Reservation and Off-Reservation Trust Land, AZ": "Hualapai Indian Reservation and Off-Reservation Trust Land, AZ",
        "Hughes ANVSA, AK": "Hughes ANVSA, AK",
        "Huron Potawatomi Reservation and Off-Reservation Trust Land, MI": "Huron Potawatomi Reservation and Off-Reservation Trust Land, MI",
        "Huslia ANVSA, AK": "Huslia ANVSA, AK",
        "Hydaburg ANVSA, AK": "Hydaburg ANVSA, AK",
        "Igiugig ANVSA, AK": "Igiugig ANVSA, AK",
        "Iliamna ANVSA, AK": "Iliamna ANVSA, AK",
        "Immokalee Reservation, FL": "Immokalee Reservation, FL",
        "Inaja and Cosmit Reservation, CA": "Inaja and Cosmit Reservation, CA",
        "Inalik ANVSA, AK": "Inalik ANVSA, AK",
        "Indian Township Reservation, ME": "Indian Township Reservation, ME",
        "Ione Band of Miwok TDSA, CA": "Ione Band of Miwok TDSA, CA",
        "Iowa (KS-NE) Reservation and Off-Reservation Trust Land, KS--NE": "Iowa (KS-NE) Reservation and Off-Reservation Trust Land, KS--NE",
        "Iowa OTSA, OK": "Iowa OTSA, OK",
        "Isabella Reservation, MI": "Isabella Reservation, MI",
        "Isleta Pueblo, NM": "Isleta Pueblo, NM",
        "Ivanof Bay ANVSA, AK": "Ivanof Bay ANVSA, AK",
        "Jackson Rancheria, CA": "Jackson Rancheria, CA",
        "Jamestown S'Klallam Reservation and Off-Reservation Trust Land, WA": "Jamestown S'Klallam Reservation and Off-Reservation Trust Land, WA",
        "Jamul Indian Village, CA": "Jamul Indian Village, CA",
        "Jemez Pueblo, NM": "Jemez Pueblo, NM",
        "Jena Band of Choctaw Reservation, LA": "Jena Band of Choctaw Reservation, LA",
        "Jicarilla Apache Nation Reservation and Off-Reservation Trust Land, NM": "Jicarilla Apache Nation Reservation and Off-Reservation Trust Land, NM",
        "Kahikinui Hawaiian Home Land, HI": "Kahikinui Hawaiian Home Land, HI",
        "Kaibab Indian Reservation, AZ": "Kaibab Indian Reservation, AZ",
        "Kakaina-Kumuhau Hawaiian Home Land, HI": "Kakaina-Kumuhau Hawaiian Home Land, HI",
        "Kake ANVSA, AK": "Kake ANVSA, AK",
        "Kaktovik ANVSA, AK": "Kaktovik ANVSA, AK",
        "Kalaeloa Hawaiian Home Land, HI": "Kalaeloa Hawaiian Home Land, HI",
        "Kalamaula Hawaiian Home Land, HI": "Kalamaula Hawaiian Home Land, HI",
        "Kalaupapa Hawaiian Home Land, HI": "Kalaupapa Hawaiian Home Land, HI",
        "Kalawahine Hawaiian Home Land, HI": "Kalawahine Hawaiian Home Land, HI",
        "Kalispel Reservation and Off-Reservation Trust Land, WA": "Kalispel Reservation and Off-Reservation Trust Land, WA",
        "Kalskag ANVSA, AK": "Kalskag ANVSA, AK",
        "Kaltag ANVSA, AK": "Kaltag ANVSA, AK",
        "Kamaoa-Puueo Hawaiian Home Land, HI": "Kamaoa-Puueo Hawaiian Home Land, HI",
        "Kamiloloa-Makakupaia Hawaiian Home Land, HI": "Kamiloloa-Makakupaia Hawaiian Home Land, HI",
        "Kamoku-Kapulena Hawaiian Home Land, HI": "Kamoku-Kapulena Hawaiian Home Land, HI",
        "Kanehili Hawaiian Home Land, HI": "Kanehili Hawaiian Home Land, HI",
        "Kaohe-Olaa Hawaiian Home Land, HI": "Kaohe-Olaa Hawaiian Home Land, HI",
        "Kapaa Hawaiian Home Land, HI": "Kapaa Hawaiian Home Land, HI",
        "Kapaakea Hawaiian Home Land, HI": "Kapaakea Hawaiian Home Land, HI",
        "Kapolei Hawaiian Home Land, HI": "Kapolei Hawaiian Home Land, HI",
        "Karluk ANVSA, AK": "Karluk ANVSA, AK",
        "Karuk Reservation and Off-Reservation Trust Land, CA": "Karuk Reservation and Off-Reservation Trust Land, CA",
        "Kasaan ANVSA, AK": "Kasaan ANVSA, AK",
        "Kasigluk ANVSA, AK": "Kasigluk ANVSA, AK",
        "Kaumana Hawaiian Home Land, HI": "Kaumana Hawaiian Home Land, HI",
        "Kaupea Hawaiian Home Land, HI": "Kaupea Hawaiian Home Land, HI",
        "Kaw OTSA, OK": "Kaw OTSA, OK",
        "Kaw/Ponca joint-use OTSA, OK": "Kaw/Ponca joint-use OTSA, OK",
        "Kawaihae Hawaiian Home Land, HI": "Kawaihae Hawaiian Home Land, HI",
        "Keahuolu Hawaiian Home Land, HI": "Keahuolu Hawaiian Home Land, HI",
        "Kealakehe Hawaiian Home Land, HI": "Kealakehe Hawaiian Home Land, HI",
        "Keanae-Wailua Hawaiian Home Land, HI": "Keanae-Wailua Hawaiian Home Land, HI",
        "Keaukaha Hawaiian Home Land, HI": "Keaukaha Hawaiian Home Land, HI",
        "Kekaha Hawaiian Home Land, HI": "Kekaha Hawaiian Home Land, HI",
        "Kenaitze ANVSA, AK": "Kenaitze ANVSA, AK",
        "Keokea (Agricultural) Hawaiian Home Land, HI": "Keokea (Agricultural) Hawaiian Home Land, HI",
        "Keoniki Hawaiian Home Land, HI": "Keoniki Hawaiian Home Land, HI",
        "Ketchikan ANVSA, AK": "Ketchikan ANVSA, AK",
        "Kewalo Hawaiian Home Land, HI": "Kewalo Hawaiian Home Land, HI",
        "Kiana ANVSA, AK": "Kiana ANVSA, AK",
        "Kickapoo (KS) Reservation, KS": "Kickapoo (KS) Reservation, KS",
        "Kickapoo (KS) Reservation/Sac and Fox Nation Trust Land joint-use area, KS": "Kickapoo (KS) Reservation/Sac and Fox Nation Trust Land joint-use area, KS",
        "Kickapoo (TX) Reservation and Off-Reservation Trust Land, TX": "Kickapoo (TX) Reservation and Off-Reservation Trust Land, TX",
        "Kickapoo OTSA, OK": "Kickapoo OTSA, OK",
        "King Cove ANVSA, AK": "King Cove ANVSA, AK",
        "King Salmon ANVSA, AK": "King Salmon ANVSA, AK",
        "Kiowa-Comanche-Apache-Fort Sill Apache OTSA, OK": "Kiowa-Comanche-Apache-Fort Sill Apache OTSA, OK",
        "Kiowa-Comanche-Apache-Ft Sill Apache/Caddo-Wichita-Delaware joint-use OTSA, OK": "Kiowa-Comanche-Apache-Ft Sill Apache/Caddo-Wichita-Delaware joint-use OTSA, OK",
        "Kipnuk ANVSA, AK": "Kipnuk ANVSA, AK",
        "Kivalina ANVSA, AK": "Kivalina ANVSA, AK",
        "Klamath Reservation, OR": "Klamath Reservation, OR",
        "Klawock ANVSA, AK": "Klawock ANVSA, AK",
        "Knik ANVSA, AK": "Knik ANVSA, AK",
        "Kobuk ANVSA, AK": "Kobuk ANVSA, AK",
        "Kodiak ANVSA, AK": "Kodiak ANVSA, AK",
        "Kokhanok ANVSA, AK": "Kokhanok ANVSA, AK",
        "Kolaoa Hawaiian Home Land, HI": "Kolaoa Hawaiian Home Land, HI",
        "Kongiganak ANVSA, AK": "Kongiganak ANVSA, AK",
        "Kootenai Reservation and Off-Reservation Trust Land, ID": "Kootenai Reservation and Off-Reservation Trust Land, ID",
        "Kotlik ANVSA, AK": "Kotlik ANVSA, AK",
        "Kotzebue ANVSA, AK": "Kotzebue ANVSA, AK",
        "Koyuk ANVSA, AK": "Koyuk ANVSA, AK",
        "Koyukuk ANVSA, AK": "Koyukuk ANVSA, AK",
        "Kwethluk ANVSA, AK": "Kwethluk ANVSA, AK",
        "Kwigillingok ANVSA, AK": "Kwigillingok ANVSA, AK",
        "Kwinhagak ANVSA, AK": "Kwinhagak ANVSA, AK",
        "L'Anse Reservation and Off-Reservation Trust Land, MI": "L'Anse Reservation and Off-Reservation Trust Land, MI",
        "La Jolla Reservation, CA": "La Jolla Reservation, CA",
        "La Posta Indian Reservation, CA": "La Posta Indian Reservation, CA",
        "Lac Courte Oreilles Reservation and Off-Reservation Trust Land, WI": "Lac Courte Oreilles Reservation and Off-Reservation Trust Land, WI",
        "Lac Vieux Desert Reservation, MI": "Lac Vieux Desert Reservation, MI",
        "Lac du Flambeau Reservation, WI": "Lac du Flambeau Reservation, WI",
        "Laguna Pueblo and Off-Reservation Trust Land, NM": "Laguna Pueblo and Off-Reservation Trust Land, NM",
        "Lake Minchumina ANVSA, AK": "Lake Minchumina ANVSA, AK",
        "Lake Traverse Reservation and Off-Reservation Trust Land, SD--ND": "Lake Traverse Reservation and Off-Reservation Trust Land, SD--ND",
        "Lalamilo Hawaiian Home Land, HI": "Lalamilo Hawaiian Home Land, HI",
        "Lanai City Hawaiian Home Land, HI": "Lanai City Hawaiian Home Land, HI",
        "Larsen Bay ANVSA, AK": "Larsen Bay ANVSA, AK",
        "Las Vegas Indian Colony, NV": "Las Vegas Indian Colony, NV",
        "Laytonville Rancheria, CA": "Laytonville Rancheria, CA",
        "Leech Lake Reservation and Off-Reservation Trust Land, MN": "Leech Lake Reservation and Off-Reservation Trust Land, MN",
        "Leialii Hawaiian Home Land, HI": "Leialii Hawaiian Home Land, HI",
        "Lenape Indian Tribe of Delaware SDTSA, DE": "Lenape Indian Tribe of Delaware SDTSA, DE",
        "Lesnoi ANVSA, AK": "Lesnoi ANVSA, AK",
        "Levelock ANVSA, AK": "Levelock ANVSA, AK",
        "Likely Rancheria, CA": "Likely Rancheria, CA",
        "Lime Village ANVSA, AK": "Lime Village ANVSA, AK",
        "Little River Reservation and Off-Reservation Trust Land, MI": "Little River Reservation and Off-Reservation Trust Land, MI",
        "Little Traverse Bay Reservation and Off-Reservation Trust Land, MI": "Little Traverse Bay Reservation and Off-Reservation Trust Land, MI",
        "Lone Pine Reservation, CA": "Lone Pine Reservation, CA",
        "Lookout Rancheria, CA": "Lookout Rancheria, CA",
        "Los Coyotes Reservation, CA": "Los Coyotes Reservation, CA",
        "Lovelock Indian Colony, NV": "Lovelock Indian Colony, NV",
        "Lower Brule Reservation and Off-Reservation Trust Land, SD": "Lower Brule Reservation and Off-Reservation Trust Land, SD",
        "Lower Elwha Reservation and Off-Reservation Trust Land, WA": "Lower Elwha Reservation and Off-Reservation Trust Land, WA",
        "Lower Kalskag ANVSA, AK": "Lower Kalskag ANVSA, AK",
        "Lower Sioux Indian Community, MN": "Lower Sioux Indian Community, MN",
        "Lualualei Hawaiian Home Land, HI": "Lualualei Hawaiian Home Land, HI",
        "Lumbee (state) SDTSA, NC": "Lumbee (state) SDTSA, NC",
        "Lummi Reservation, WA": "Lummi Reservation, WA",
        "Lytton Rancheria, CA": "Lytton Rancheria, CA",
        "MOWA Choctaw Reservation (state), AL": "MOWA Choctaw Reservation (state), AL",
        "MaChis Lower Creek (state) SDTSA, AL": "MaChis Lower Creek (state) SDTSA, AL",
        "Maili Hawaiian Home Land, HI": "Maili Hawaiian Home Land, HI",
        "Makah Indian Reservation, WA": "Makah Indian Reservation, WA",
        "Makaha Valley Hawaiian Home Land, HI": "Makaha Valley Hawaiian Home Land, HI",
        "Makuu Hawaiian Home Land, HI": "Makuu Hawaiian Home Land, HI",
        "Maluohai Hawaiian Home Land, HI": "Maluohai Hawaiian Home Land, HI",
        "Manchester-Point Arena Rancheria, CA": "Manchester-Point Arena Rancheria, CA",
        "Manley Hot Springs ANVSA, AK": "Manley Hot Springs ANVSA, AK",
        "Manokotak ANVSA, AK": "Manokotak ANVSA, AK",
        "Manzanita Reservation and Off-Reservation Trust Land, CA": "Manzanita Reservation and Off-Reservation Trust Land, CA",
        "Maricopa (Ak Chin) Indian Reservation and Off-Reservation Trust Land, AZ": "Maricopa (Ak Chin) Indian Reservation and Off-Reservation Trust Land, AZ",
        "Marshall ANVSA, AK": "Marshall ANVSA, AK",
        "Mary's Igloo ANVSA, AK": "Mary's Igloo ANVSA, AK",
        "Mashantucket Pequot Reservation, CT": "Mashantucket Pequot Reservation, CT",
        "Mashpee Wampanoag Trust Land, MA": "Mashpee Wampanoag Trust Land, MA",
        "Match-e-be-nash-she-wish Band of Pottawatomi Reservation and Off-Reservation Trust Land, MI": "Match-e-be-nash-she-wish Band of Pottawatomi Reservation and Off-Reservation Trust Land, MI",
        "Mattaponi Reservation (state), VA": "Mattaponi Reservation (state), VA",
        "McGrath ANVSA, AK": "McGrath ANVSA, AK",
        "Mechoopda TDSA, CA": "Mechoopda TDSA, CA",
        "Meherrin (state) SDTSA, NC": "Meherrin (state) SDTSA, NC",
        "Mekoryuk ANVSA, AK": "Mekoryuk ANVSA, AK",
        "Menominee Reservation, WI": "Menominee Reservation, WI",
        "Mentasta Lake ANVSA, AK": "Mentasta Lake ANVSA, AK",
        "Mesa Grande Reservation, CA": "Mesa Grande Reservation, CA",
        "Mescalero Reservation, NM": "Mescalero Reservation, NM",
        "Miami OTSA, OK": "Miami OTSA, OK",
        "Miami/Peoria joint-use OTSA, OK": "Miami/Peoria joint-use OTSA, OK",
        "Miccosukee Reservation and Off-Reservation Trust Land, FL": "Miccosukee Reservation and Off-Reservation Trust Land, FL",
        "Middletown Rancheria, CA": "Middletown Rancheria, CA",
        "Mille Lacs Reservation and Off-Reservation Trust Land, MN": "Mille Lacs Reservation and Off-Reservation Trust Land, MN",
        "Minnesota Chippewa Trust Land, MN": "Minnesota Chippewa Trust Land, MN",
        "Minto ANVSA, AK": "Minto ANVSA, AK",
        "Mississippi Choctaw Reservation, MS": "Mississippi Choctaw Reservation, MS",
        "Moapa River Indian Reservation, NV": "Moapa River Indian Reservation, NV",
        "Modoc OTSA, OK": "Modoc OTSA, OK",
        "Mohegan Reservation and Off-Reservation Trust Land, CT": "Mohegan Reservation and Off-Reservation Trust Land, CT",
        "Moloaa Hawaiian Home Land, HI": "Moloaa Hawaiian Home Land, HI",
        "Montgomery Creek Rancheria, CA": "Montgomery Creek Rancheria, CA",
        "Mooretown Rancheria and Off-Reservation Trust Land, CA": "Mooretown Rancheria and Off-Reservation Trust Land, CA",
        "Morongo Reservation and Off-Reservation Trust Land, CA": "Morongo Reservation and Off-Reservation Trust Land, CA",
        "Mountain Village ANVSA, AK": "Mountain Village ANVSA, AK",
        "Muckleshoot Reservation and Off-Reservation Trust Land, WA": "Muckleshoot Reservation and Off-Reservation Trust Land, WA",
        "Naknek ANVSA, AK": "Naknek ANVSA, AK",
        "Nambe Pueblo and Off-Reservation Trust Land, NM": "Nambe Pueblo and Off-Reservation Trust Land, NM",
        "Nanakuli Hawaiian Home Land, HI": "Nanakuli Hawaiian Home Land, HI",
        "Nanticoke Indian Tribe (state) SDTSA, DE": "Nanticoke Indian Tribe (state) SDTSA, DE",
        "Nanticoke Lenni Lenape (state) SDTSA, NJ": "Nanticoke Lenni Lenape (state) SDTSA, NJ",
        "Nanwalek ANVSA, AK": "Nanwalek ANVSA, AK",
        "Napaimute ANVSA, AK": "Napaimute ANVSA, AK",
        "Napakiak ANVSA, AK": "Napakiak ANVSA, AK",
        "Napaskiak ANVSA, AK": "Napaskiak ANVSA, AK",
        "Narragansett Reservation, RI": "Narragansett Reservation, RI",
        "Navajo Nation Reservation and Off-Reservation Trust Land, AZ--NM--UT": "Navajo Nation Reservation and Off-Reservation Trust Land, AZ--NM--UT",
        "Nelson Lagoon ANVSA, AK": "Nelson Lagoon ANVSA, AK",
        "Nenana ANVSA, AK": "Nenana ANVSA, AK",
        "New Koliganek ANVSA, AK": "New Koliganek ANVSA, AK",
        "New Stuyahok ANVSA, AK": "New Stuyahok ANVSA, AK",
        "Newhalen ANVSA, AK": "Newhalen ANVSA, AK",
        "Newtok ANVSA, AK": "Newtok ANVSA, AK",
        "Nez Perce Reservation, ID": "Nez Perce Reservation, ID",
        "Nienie Hawaiian Home Land, HI": "Nienie Hawaiian Home Land, HI",
        "Nightmute ANVSA, AK": "Nightmute ANVSA, AK",
        "Nikolai ANVSA, AK": "Nikolai ANVSA, AK",
        "Nikolski ANVSA, AK": "Nikolski ANVSA, AK",
        "Ninilchik ANVSA, AK": "Ninilchik ANVSA, AK",
        "Nisqually Reservation, WA": "Nisqually Reservation, WA",
        "Noatak ANVSA, AK": "Noatak ANVSA, AK",
        "Nome ANVSA, AK": "Nome ANVSA, AK",
        "Nondalton ANVSA, AK": "Nondalton ANVSA, AK",
        "Nooksack Reservation and Off-Reservation Trust Land, WA": "Nooksack Reservation and Off-Reservation Trust Land, WA",
        "Noorvik ANVSA, AK": "Noorvik ANVSA, AK",
        "North Fork Rancheria and Off-Reservation Trust Land, CA": "North Fork Rancheria and Off-Reservation Trust Land, CA",
        "Northern Cheyenne Indian Reservation and Off-Reservation Trust Land, MT--SD": "Northern Cheyenne Indian Reservation and Off-Reservation Trust Land, MT--SD",
        "Northway ANVSA, AK": "Northway ANVSA, AK",
        "Northwestern Shoshone Reservation, UT": "Northwestern Shoshone Reservation, UT",
        "Nuiqsut ANVSA, AK": "Nuiqsut ANVSA, AK",
        "Nulato ANVSA, AK": "Nulato ANVSA, AK",
        "Nunam Iqua ANVSA, AK": "Nunam Iqua ANVSA, AK",
        "Nunapitchuk ANVSA, AK": "Nunapitchuk ANVSA, AK",
        "Occaneechi-Saponi SDTSA, NC": "Occaneechi-Saponi SDTSA, NC",
        "Ohkay Owingeh, NM": "Ohkay Owingeh, NM",
        "Ohogamiut ANVSA, AK": "Ohogamiut ANVSA, AK",
        "Oil Springs Reservation, NY": "Oil Springs Reservation, NY",
        "Old Harbor ANVSA, AK": "Old Harbor ANVSA, AK",
        "Omaha Reservation, NE--IA": "Omaha Reservation, NE--IA",
        "Oneida (WI) Reservation and Off-Reservation Trust Land, WI": "Oneida (WI) Reservation and Off-Reservation Trust Land, WI",
        "Oneida Nation Reservation, NY": "Oneida Nation Reservation, NY",
        "Onondaga Nation Reservation, NY": "Onondaga Nation Reservation, NY",
        "Ontonagon Reservation, MI": "Ontonagon Reservation, MI",
        "Osage Reservation, OK": "Osage Reservation, OK",
        "Oscarville ANVSA, AK": "Oscarville ANVSA, AK",
        "Otoe-Missouria OTSA, OK": "Otoe-Missouria OTSA, OK",
        "Ottawa OTSA, OK": "Ottawa OTSA, OK",
        "Ouzinkie ANVSA, AK": "Ouzinkie ANVSA, AK",
        "Paimiut ANVSA, AK": "Paimiut ANVSA, AK",
        "Paiute (UT) Reservation, UT": "Paiute (UT) Reservation, UT",
        "Pala Reservation, CA": "Pala Reservation, CA",
        "Pamunkey Reservation (state), VA": "Pamunkey Reservation (state), VA",
        "Panaewa (Agricultural) Hawaiian Home Land, HI": "Panaewa (Agricultural) Hawaiian Home Land, HI",
        "Panaewa (Residential) Hawaiian Home Land, HI": "Panaewa (Residential) Hawaiian Home Land, HI",
        "Papakolea Hawaiian Home Land, HI": "Papakolea Hawaiian Home Land, HI",
        "Pascua Pueblo Yaqui Reservation and Off-Reservation Trust Land, AZ": "Pascua Pueblo Yaqui Reservation and Off-Reservation Trust Land, AZ",
        "Paskenta Rancheria, CA": "Paskenta Rancheria, CA",
        "Passamaquoddy Trust Land, ME": "Passamaquoddy Trust Land, ME",
        "Pauahi Hawaiian Home Land, HI": "Pauahi Hawaiian Home Land, HI",
        "Paucatuck Eastern Pequot Reservation (state), CT": "Paucatuck Eastern Pequot Reservation (state), CT",
        "Paukukalo Hawaiian Home Land, HI": "Paukukalo Hawaiian Home Land, HI",
        "Pauma and Yuima Reservation, CA": "Pauma and Yuima Reservation, CA",
        "Pawnee OTSA, OK": "Pawnee OTSA, OK",
        "Pearl City Hawaiian Home Land, HI": "Pearl City Hawaiian Home Land, HI",
        "Pechanga Reservation, CA": "Pechanga Reservation, CA",
        "Pedro Bay ANVSA, AK": "Pedro Bay ANVSA, AK",
        "Pee Dee SDTSA, SC": "Pee Dee SDTSA, SC",
        "Penobscot Reservation and Off-Reservation Trust Land, ME": "Penobscot Reservation and Off-Reservation Trust Land, ME",
        "Peoria OTSA, OK": "Peoria OTSA, OK",
        "Perryville ANVSA, AK": "Perryville ANVSA, AK",
        "Petersburg ANVSA, AK": "Petersburg ANVSA, AK",
        "Picayune Rancheria and Off-Reservation Trust Land, CA": "Picayune Rancheria and Off-Reservation Trust Land, CA",
        "Picuris Pueblo, NM": "Picuris Pueblo, NM",
        "Piihonua Hawaiian Home Land, HI": "Piihonua Hawaiian Home Land, HI",
        "Pilot Point ANVSA, AK": "Pilot Point ANVSA, AK",
        "Pilot Station ANVSA, AK": "Pilot Station ANVSA, AK",
        "Pine Ridge Reservation, SD--NE": "Pine Ridge Reservation, SD--NE",
        "Pinoleville Rancheria, CA": "Pinoleville Rancheria, CA",
        "Pit River Trust Land, CA": "Pit River Trust Land, CA",
        "Pitkas Point ANVSA, AK": "Pitkas Point ANVSA, AK",
        "Platinum ANVSA, AK": "Platinum ANVSA, AK",
        "Pleasant Point Reservation, ME": "Pleasant Point Reservation, ME",
        "Poarch Creek Reservation and Off-Reservation Trust Land, AL--FL": "Poarch Creek Reservation and Off-Reservation Trust Land, AL--FL",
        "Point Hope ANVSA, AK": "Point Hope ANVSA, AK",
        "Point Lay ANVSA, AK": "Point Lay ANVSA, AK",
        "Pokagon Reservation and Off-Reservation Trust Land, MI": "Pokagon Reservation and Off-Reservation Trust Land, MI",
        "Ponca (NE) Trust Land, NE--IA": "Ponca (NE) Trust Land, NE--IA",
        "Ponca OTSA, OK": "Ponca OTSA, OK",
        "Ponohawaii Hawaiian Home Land, HI": "Ponohawaii Hawaiian Home Land, HI",
        "Poospatuck Reservation (state), NY": "Poospatuck Reservation (state), NY",
        "Port Alsworth ANVSA, AK": "Port Alsworth ANVSA, AK",
        "Port Gamble Reservation and Off-Reservation Trust Land, WA": "Port Gamble Reservation and Off-Reservation Trust Land, WA",
        "Port Graham ANVSA, AK": "Port Graham ANVSA, AK",
        "Port Heiden ANVSA, AK": "Port Heiden ANVSA, AK",
        "Port Lions ANVSA, AK": "Port Lions ANVSA, AK",
        "Port Madison Reservation, WA": "Port Madison Reservation, WA",
        "Portage Creek ANVSA, AK": "Portage Creek ANVSA, AK",
        "Prairie Band of Potawatomi Nation Reservation, KS": "Prairie Band of Potawatomi Nation Reservation, KS",
        "Prairie Island Indian Community and Off-Reservation Trust Land, MN": "Prairie Island Indian Community and Off-Reservation Trust Land, MN",
        "Princess Kahanu Estates Hawaiian Home Land, HI": "Princess Kahanu Estates Hawaiian Home Land, HI",
        "Pueblo de Cochiti, NM": "Pueblo de Cochiti, NM",
        "Pueblo of Pojoaque and Off-Reservation Trust Land, NM": "Pueblo of Pojoaque and Off-Reservation Trust Land, NM",
        "Pulehunui Hawaiian Home Land, HI": "Pulehunui Hawaiian Home Land, HI",
        "Puukapu Hawaiian Home Land, HI": "Puukapu Hawaiian Home Land, HI",
        "Puyallup Reservation and Off-Reservation Trust Land, WA": "Puyallup Reservation and Off-Reservation Trust Land, WA",
        "Pyramid Lake Paiute Reservation, NV": "Pyramid Lake Paiute Reservation, NV",
        "Quapaw OTSA, OK": "Quapaw OTSA, OK",
        "Quartz Valley Reservation and Off-Reservation Trust Land, CA": "Quartz Valley Reservation and Off-Reservation Trust Land, CA",
        "Quileute Reservation, WA": "Quileute Reservation, WA",
        "Quinault Reservation, WA": "Quinault Reservation, WA",
        "Ramapough (state) SDTSA, NJ": "Ramapough (state) SDTSA, NJ",
        "Ramona Village, CA": "Ramona Village, CA",
        "Rampart ANVSA, AK": "Rampart ANVSA, AK",
        "Red Cliff Reservation and Off-Reservation Trust Land, WI": "Red Cliff Reservation and Off-Reservation Trust Land, WI",
        "Red Devil ANVSA, AK": "Red Devil ANVSA, AK",
        "Red Lake Reservation, MN": "Red Lake Reservation, MN",
        "Redding Rancheria, CA": "Redding Rancheria, CA",
        "Redwood Valley Rancheria, CA": "Redwood Valley Rancheria, CA",
        "Reno-Sparks Indian Colony and Off-Reservation Trust Land, NV": "Reno-Sparks Indian Colony and Off-Reservation Trust Land, NV",
        "Resighini Rancheria, CA": "Resighini Rancheria, CA",
        "Rincon Reservation and Off-Reservation Trust Land, CA": "Rincon Reservation and Off-Reservation Trust Land, CA",
        "Roaring Creek Rancheria, CA": "Roaring Creek Rancheria, CA",
        "Robinson Rancheria and Off-Reservation Trust Land, CA": "Robinson Rancheria and Off-Reservation Trust Land, CA",
        "Rocky Boy's Reservation and Off-Reservation Trust Land, MT": "Rocky Boy's Reservation and Off-Reservation Trust Land, MT",
        "Rohnerville Rancheria, CA": "Rohnerville Rancheria, CA",
        "Rosebud Indian Reservation and Off-Reservation Trust Land, SD": "Rosebud Indian Reservation and Off-Reservation Trust Land, SD",
        "Round Valley Reservation and Off-Reservation Trust Land, CA": "Round Valley Reservation and Off-Reservation Trust Land, CA",
        "Ruby ANVSA, AK": "Ruby ANVSA, AK",
        "Rumsey Indian Rancheria, CA": "Rumsey Indian Rancheria, CA",
        "Russian Mission ANVSA, AK": "Russian Mission ANVSA, AK",
        "Sac and Fox Nation Reservation and Off-Reservation Trust Land, NE--KS": "Sac and Fox Nation Reservation and Off-Reservation Trust Land, NE--KS",
        "Sac and Fox OTSA, OK": "Sac and Fox OTSA, OK",
        "Sac and Fox/Meskwaki Settlement and Off-Reservation Trust Land, IA": "Sac and Fox/Meskwaki Settlement and Off-Reservation Trust Land, IA",
        "Saint Croix Reservation and Off-Reservation Trust Land, WI": "Saint Croix Reservation and Off-Reservation Trust Land, WI",
        "Saint Regis Mohawk Reservation, NY": "Saint Regis Mohawk Reservation, NY",
        "Salamatof ANVSA, AK": "Salamatof ANVSA, AK",
        "Salt River Reservation, AZ": "Salt River Reservation, AZ",
        "Samish TDSA, WA": "Samish TDSA, WA",
        "San Carlos Reservation, AZ": "San Carlos Reservation, AZ",
        "San Felipe Pueblo, NM": "San Felipe Pueblo, NM",
        "San Felipe Pueblo/Santa Ana Pueblo joint-use area, NM": "San Felipe Pueblo/Santa Ana Pueblo joint-use area, NM",
        "San Felipe Pueblo/Santo Domingo Pueblo joint-use area, NM": "San Felipe Pueblo/Santo Domingo Pueblo joint-use area, NM",
        "San Ildefonso Pueblo and Off-Reservation Trust Land, NM": "San Ildefonso Pueblo and Off-Reservation Trust Land, NM",
        "San Manuel Reservation and Off-Reservation Trust Land, CA": "San Manuel Reservation and Off-Reservation Trust Land, CA",
        "San Pasqual Reservation and Off-Reservation Trust Land, CA": "San Pasqual Reservation and Off-Reservation Trust Land, CA",
        "Sand Point ANVSA, AK": "Sand Point ANVSA, AK",
        "Sandia Pueblo, NM": "Sandia Pueblo, NM",
        "Santa Ana Pueblo, NM": "Santa Ana Pueblo, NM",
        "Santa Clara Pueblo and Off-Reservation Trust Land, NM": "Santa Clara Pueblo and Off-Reservation Trust Land, NM",
        "Santa Rosa Rancheria, CA": "Santa Rosa Rancheria, CA",
        "Santa Rosa Reservation, CA": "Santa Rosa Reservation, CA",
        "Santa Ynez Reservation, CA": "Santa Ynez Reservation, CA",
        "Santa Ysabel Reservation, CA": "Santa Ysabel Reservation, CA",
        "Santee Reservation, NE": "Santee Reservation, NE",
        "Santee SDTSA, SC": "Santee SDTSA, SC",
        "Santo Domingo Pueblo, NM": "Santo Domingo Pueblo, NM",
        "Sappony SDTSA, NC": "Sappony SDTSA, NC",
        "Sauk-Suiattle Reservation, WA": "Sauk-Suiattle Reservation, WA",
        "Sault Sainte Marie Reservation and Off-Reservation Trust Land, MI": "Sault Sainte Marie Reservation and Off-Reservation Trust Land, MI",
        "Savoonga ANVSA, AK": "Savoonga ANVSA, AK",
        "Saxman ANVSA, AK": "Saxman ANVSA, AK",
        "Scammon Bay ANVSA, AK": "Scammon Bay ANVSA, AK",
        "Schaghticoke Reservation (state), CT": "Schaghticoke Reservation (state), CT",
        "Selawik ANVSA, AK": "Selawik ANVSA, AK",
        "Seldovia ANVSA, AK": "Seldovia ANVSA, AK",
        "Seminole (FL) Trust Land, FL": "Seminole (FL) Trust Land, FL",
        "Seminole OTSA, OK": "Seminole OTSA, OK",
        "Seneca-Cayuga OTSA, OK": "Seneca-Cayuga OTSA, OK",
        "Shageluk ANVSA, AK": "Shageluk ANVSA, AK",
        "Shakopee Mdewakanton Sioux Community and Off-Reservation Trust Land, MN": "Shakopee Mdewakanton Sioux Community and Off-Reservation Trust Land, MN",
        "Shaktoolik ANVSA, AK": "Shaktoolik ANVSA, AK",
        "Sherwood Valley Rancheria and Off-Reservation Trust Land, CA": "Sherwood Valley Rancheria and Off-Reservation Trust Land, CA",
        "Shingle Springs Rancheria and Off-Reservation Trust Land, CA": "Shingle Springs Rancheria and Off-Reservation Trust Land, CA",
        "Shinnecock Reservation (state), NY": "Shinnecock Reservation (state), NY",
        "Shishmaref ANVSA, AK": "Shishmaref ANVSA, AK",
        "Shoalwater Bay Indian Reservation and Off-Reservation Trust Land, WA": "Shoalwater Bay Indian Reservation and Off-Reservation Trust Land, WA",
        "Shungnak ANVSA, AK": "Shungnak ANVSA, AK",
        "Siletz Reservation and Off-Reservation Trust Land, OR": "Siletz Reservation and Off-Reservation Trust Land, OR",
        "Sitka ANVSA, AK": "Sitka ANVSA, AK",
        "Skagway ANVSA, AK": "Skagway ANVSA, AK",
        "Skokomish Reservation and Off-Reservation Trust Land, WA": "Skokomish Reservation and Off-Reservation Trust Land, WA",
        "Skull Valley Reservation, UT": "Skull Valley Reservation, UT",
        "Sleetmute ANVSA, AK": "Sleetmute ANVSA, AK",
        "Smith River Reservation and Off-Reservation Trust Land, CA": "Smith River Reservation and Off-Reservation Trust Land, CA",
        "Snoqualmie Reservation, WA": "Snoqualmie Reservation, WA",
        "Soboba Reservation and Off-Reservation Trust Land, CA": "Soboba Reservation and Off-Reservation Trust Land, CA",
        "Sokaogon Chippewa Community, WI": "Sokaogon Chippewa Community, WI",
        "Solomon ANVSA, AK": "Solomon ANVSA, AK",
        "South Fork Reservation and Off-Reservation Trust Land, NV": "South Fork Reservation and Off-Reservation Trust Land, NV",
        "South Maui Hawaiian Home Land, HI": "South Maui Hawaiian Home Land, HI",
        "South Naknek ANVSA, AK": "South Naknek ANVSA, AK",
        "Southern Ute Reservation, CO": "Southern Ute Reservation, CO",
        "Spirit Lake Reservation, ND": "Spirit Lake Reservation, ND",
        "Spokane Reservation and Off-Reservation Trust Land, WA": "Spokane Reservation and Off-Reservation Trust Land, WA",
        "Squaxin Island Reservation and Off-Reservation Trust Land, WA": "Squaxin Island Reservation and Off-Reservation Trust Land, WA",
        "St. George ANVSA, AK": "St. George ANVSA, AK",
        "St. Michael ANVSA, AK": "St. Michael ANVSA, AK",
        "St. Paul ANVSA, AK": "St. Paul ANVSA, AK",
        "Standing Rock Reservation, SD--ND": "Standing Rock Reservation, SD--ND",
        "Star Muskogee Creek (state) SDTSA, AL": "Star Muskogee Creek (state) SDTSA, AL",
        "Stebbins ANVSA, AK": "Stebbins ANVSA, AK",
        "Stevens Village ANVSA, AK": "Stevens Village ANVSA, AK",
        "Stewart Community, NV": "Stewart Community, NV",
        "Stewarts Point Rancheria and Off-Reservation Trust Land, CA": "Stewarts Point Rancheria and Off-Reservation Trust Land, CA",
        "Stillaguamish Reservation and Off-Reservation Trust Land, WA": "Stillaguamish Reservation and Off-Reservation Trust Land, WA",
        "Stockbridge Munsee Community and Off-Reservation Trust Land, WI": "Stockbridge Munsee Community and Off-Reservation Trust Land, WI",
        "Stony River ANVSA, AK": "Stony River ANVSA, AK",
        "Sulphur Bank Rancheria, CA": "Sulphur Bank Rancheria, CA",
        "Summit Lake Reservation and Off-Reservation Trust Land, NV": "Summit Lake Reservation and Off-Reservation Trust Land, NV",
        "Susanville Indian Rancheria and Off-Reservation Trust Land, CA": "Susanville Indian Rancheria and Off-Reservation Trust Land, CA",
        "Swinomish Reservation and Off-Reservation Trust Land, WA": "Swinomish Reservation and Off-Reservation Trust Land, WA",
        "Sycuan Reservation and Off-Reservation Trust Land, CA": "Sycuan Reservation and Off-Reservation Trust Land, CA",
        "Table Bluff Reservation, CA": "Table Bluff Reservation, CA",
        "Table Mountain Rancheria, CA": "Table Mountain Rancheria, CA",
        "Takotna ANVSA, AK": "Takotna ANVSA, AK",
        "Tama Reservation (state), GA": "Tama Reservation (state), GA",
        "Tampa Reservation, FL": "Tampa Reservation, FL",
        "Tanacross ANVSA, AK": "Tanacross ANVSA, AK",
        "Tanana ANVSA, AK": "Tanana ANVSA, AK",
        "Taos Pueblo and Off-Reservation Trust Land, NM": "Taos Pueblo and Off-Reservation Trust Land, NM",
        "Tatitlek ANVSA, AK": "Tatitlek ANVSA, AK",
        "Tazlina ANVSA, AK": "Tazlina ANVSA, AK",
        "Telida ANVSA, AK": "Telida ANVSA, AK",
        "Teller ANVSA, AK": "Teller ANVSA, AK",
        "Tesuque Pueblo and Off-Reservation Trust Land, NM": "Tesuque Pueblo and Off-Reservation Trust Land, NM",
        "Tetlin ANVSA, AK": "Tetlin ANVSA, AK",
        "Timbi-Sha Shoshone Reservation and Off-Reservation Trust Land, CA--NV": "Timbi-Sha Shoshone Reservation and Off-Reservation Trust Land, CA--NV",
        "Togiak ANVSA, AK": "Togiak ANVSA, AK",
        "Tohono O'odham Nation Reservation and Off-Reservation Trust Land, AZ": "Tohono O'odham Nation Reservation and Off-Reservation Trust Land, AZ",
        "Toksook Bay ANVSA, AK": "Toksook Bay ANVSA, AK",
        "Tonawanda Reservation, NY": "Tonawanda Reservation, NY",
        "Tonkawa OTSA, OK": "Tonkawa OTSA, OK",
        "Tonto Apache Reservation and Off-Reservation Trust Land, AZ": "Tonto Apache Reservation and Off-Reservation Trust Land, AZ",
        "Torres-Martinez Reservation, CA": "Torres-Martinez Reservation, CA",
        "Trinidad Rancheria and Off-Reservation Trust Land, CA": "Trinidad Rancheria and Off-Reservation Trust Land, CA",
        "Tulalip Reservation and Off-Reservation Trust Land, WA": "Tulalip Reservation and Off-Reservation Trust Land, WA",
        "Tule River Reservation and Off-Reservation Trust Land, CA": "Tule River Reservation and Off-Reservation Trust Land, CA",
        "Tuluksak ANVSA, AK": "Tuluksak ANVSA, AK",
        "Tunica-Biloxi Reservation and Off-Reservation Trust Land, LA": "Tunica-Biloxi Reservation and Off-Reservation Trust Land, LA",
        "Tuntutuliak ANVSA, AK": "Tuntutuliak ANVSA, AK",
        "Tununak ANVSA, AK": "Tununak ANVSA, AK",
        "Tuolumne Rancheria, CA": "Tuolumne Rancheria, CA",
        "Turtle Mountain Reservation and Off-Reservation Trust Land, MT--ND--SD": "Turtle Mountain Reservation and Off-Reservation Trust Land, MT--ND--SD",
        "Tuscarora Nation Reservation, NY": "Tuscarora Nation Reservation, NY",
        "Twenty-Nine Palms Reservation and Off-Reservation Trust Land, CA": "Twenty-Nine Palms Reservation and Off-Reservation Trust Land, CA",
        "Twin Hills ANVSA, AK": "Twin Hills ANVSA, AK",
        "Tyonek ANVSA, AK": "Tyonek ANVSA, AK",
        "Ualapue Hawaiian Home Land, HI": "Ualapue Hawaiian Home Land, HI",
        "Ugashik ANVSA, AK": "Ugashik ANVSA, AK",
        "Uintah and Ouray Reservation and Off-Reservation Trust Land, UT": "Uintah and Ouray Reservation and Off-Reservation Trust Land, UT",
        "Umatilla Reservation and Off-Reservation Trust Land, OR": "Umatilla Reservation and Off-Reservation Trust Land, OR",
        "Unalakleet ANVSA, AK": "Unalakleet ANVSA, AK",
        "Unalaska ANVSA, AK": "Unalaska ANVSA, AK",
        "United Cherokee Ani-Yun-Wiya Nation SDTSA, AL": "United Cherokee Ani-Yun-Wiya Nation SDTSA, AL",
        "United Houma Nation (state) SDTSA, LA": "United Houma Nation (state) SDTSA, LA",
        "Upolu Hawaiian Home Land, HI": "Upolu Hawaiian Home Land, HI",
        "Upper Lake Rancheria, CA": "Upper Lake Rancheria, CA",
        "Upper Sioux Community and Off-Reservation Trust Land, MN": "Upper Sioux Community and Off-Reservation Trust Land, MN",
        "Upper Skagit Reservation and Off-Reservation Trust Land, WA": "Upper Skagit Reservation and Off-Reservation Trust Land, WA",
        "Upper South Carolina Pee Dee SDTSA, SC": "Upper South Carolina Pee Dee SDTSA, SC",
        "Ute Mountain Reservation and Off-Reservation Trust Land, CO--NM--UT": "Ute Mountain Reservation and Off-Reservation Trust Land, CO--NM--UT",
        "Venetie ANVSA, AK": "Venetie ANVSA, AK",
        "Viejas Reservation and Off-Reservation Trust Land, CA": "Viejas Reservation and Off-Reservation Trust Land, CA",
        "Waccamaw SDTSA, SC": "Waccamaw SDTSA, SC",
        "Waccamaw Siouan (state) SDTSA, NC": "Waccamaw Siouan (state) SDTSA, NC",
        "Waiahole Hawaiian Home Land, HI": "Waiahole Hawaiian Home Land, HI",
        "Waiakea Hawaiian Home Land, HI": "Waiakea Hawaiian Home Land, HI",
        "Waianae Hawaiian Home Land, HI": "Waianae Hawaiian Home Land, HI",
        "Waianae Kai Hawaiian Home Land, HI": "Waianae Kai Hawaiian Home Land, HI",
        "Waiehu Hawaiian Home Land, HI": "Waiehu Hawaiian Home Land, HI",
        "Waiku-Hana Hawaiian Home Land, HI": "Waiku-Hana Hawaiian Home Land, HI",
        "Wailau Hawaiian Home Land, HI": "Wailau Hawaiian Home Land, HI",
        "Wailua Hawaiian Home Land, HI": "Wailua Hawaiian Home Land, HI",
        "Waimanalo Hawaiian Home Land, HI": "Waimanalo Hawaiian Home Land, HI",
        "Waimanu Hawaiian Home Land, HI": "Waimanu Hawaiian Home Land, HI",
        "Waimea Hawaiian Home Land, HI": "Waimea Hawaiian Home Land, HI",
        "Wainwright ANVSA, AK": "Wainwright ANVSA, AK",
        "Waiohinu Hawaiian Home Land, HI": "Waiohinu Hawaiian Home Land, HI",
        "Waiohuli (Residential) Hawaiian Home Land, HI": "Waiohuli (Residential) Hawaiian Home Land, HI",
        "Wales ANVSA, AK": "Wales ANVSA, AK",
        "Walker River Reservation, NV": "Walker River Reservation, NV",
        "Wampanoag-Aquinnah Trust Land, MA": "Wampanoag-Aquinnah Trust Land, MA",
        "Warm Springs Reservation and Off-Reservation Trust Land, OR": "Warm Springs Reservation and Off-Reservation Trust Land, OR",
        "Washoe Ranches Trust Land, NV--CA": "Washoe Ranches Trust Land, NV--CA",
        "Wassamasaw SDTSA, SC": "Wassamasaw SDTSA, SC",
        "Wells Colony, NV": "Wells Colony, NV",
        "White Earth Reservation and Off-Reservation Trust Land, MN": "White Earth Reservation and Off-Reservation Trust Land, MN",
        "White Mountain ANVSA, AK": "White Mountain ANVSA, AK",
        "Wind River Reservation and Off-Reservation Trust Land, WY": "Wind River Reservation and Off-Reservation Trust Land, WY",
        "Winnebago Reservation and Off-Reservation Trust Land, NE--IA": "Winnebago Reservation and Off-Reservation Trust Land, NE--IA",
        "Winnemucca Indian Colony, NV": "Winnemucca Indian Colony, NV",
        "Woodfords Community, CA": "Woodfords Community, CA",
        "Wrangell ANVSA, AK": "Wrangell ANVSA, AK",
        "Wyandotte OTSA, OK": "Wyandotte OTSA, OK",
        "XL Ranch Rancheria, CA": "XL Ranch Rancheria, CA",
        "Yakama Nation Reservation and Off-Reservation Trust Land, WA": "Yakama Nation Reservation and Off-Reservation Trust Land, WA",
        "Yakutat ANVSA, AK": "Yakutat ANVSA, AK",
        "Yankton Reservation, SD": "Yankton Reservation, SD",
        "Yavapai-Apache Nation Reservation and Off-Reservation Trust Land, AZ": "Yavapai-Apache Nation Reservation and Off-Reservation Trust Land, AZ",
        "Yavapai-Prescott Reservation, AZ": "Yavapai-Prescott Reservation, AZ",
        "Yerington Colony, NV": "Yerington Colony, NV",
        "Yomba Reservation, NV": "Yomba Reservation, NV",
        "Ysleta del Sur Pueblo and Off-Reservation Trust Land, TX": "Ysleta del Sur Pueblo and Off-Reservation Trust Land, TX",
        "Yurok Reservation, CA": "Yurok Reservation, CA",
        "Zia Pueblo and Off-Reservation Trust Land, NM": "Zia Pueblo and Off-Reservation Trust Land, NM",
        "Zuni Reservation and Off-Reservation Trust Land, NM--AZ": "Zuni Reservation and Off-Reservation Trust Land, NM--AZ",
    }


def format_percentage_frame(frame: pd.DataFrame, threshhold: float = 0.01) -> Styler:
    """Adds a nice green and formats numbers as percentages"""
    percentage_columns = [col for col in frame.columns if "total" not in col]
    total_columns = [col for col in frame.columns if "total" in col]
    speaker_cols = [col for col in percentage_columns if "speaker" in col]
    styled_frame = frame.style.applymap(
        lambda v: "background-color: #e6ffe6;" if v > 0.01 else None,
        subset=speaker_cols,
    )

    return (
        styled_frame.format("{:.2%}".format, subset=percentage_columns)
        .format("{:,}".format, subset=total_columns)
        .set_sticky(axis=1)
    )


def get_percentages(frame: pd.DataFrame) -> pd.DataFrame:
    divisor_col = frame.columns[0]
    return frame[[col for col in frame.columns if col != divisor_col]].truediv(
        frame[divisor_col], axis=0
    )


def get_total_population() -> pd.DataFrame:
    return get_frame_for_county_vars(TotalPopulationVars)


def get_county_language_data() -> pd.DataFrame:
    return get_percentages(get_frame_for_county_vars(LanguageVars))


def get_public_assistance_data() -> pd.DataFrame:
    return get_percentages(get_frame_for_county_vars(PublicAssistanceVars))


def get_poverty_level_data() -> pd.DataFrame:
    """Sum the population under 185% of the poverty level and convert to a percentage"""
    poverty_level = get_frame_for_county_vars(PovertyLevelVars)
    non_total_cols = [
        col for col in poverty_level.columns if not col.startswith("total")
    ]
    poverty_level["under 185%"] = poverty_level[non_total_cols].sum(axis=1)
    poverty_level.drop(columns=non_total_cols, inplace=True)
    return get_percentages(poverty_level)


def get_county_census_data() -> pd.DataFrame:
    return (
        get_total_population()
        .join(get_public_assistance_data())
        .join(get_poverty_level_data())
        .join(get_county_language_data())
    )


def get_wic_coverage_frame(state_name: str) -> pd.DataFrame:
    frame = pd.read_excel(
        "./data/wic-coverage-rates-by-state-2018.xlsx",
        sheet_name="Coverage Rate by State",
        index_col=0,
    )
    frame = frame[frame.index == state_name]
    percentage_columns = [col for col in frame.columns if "Number" not in col]
    total_columns = [col for col in frame.columns if "Number" in col]
    return frame.style.format("{:.0%}".format, subset=percentage_columns).format(
        "{:,.0f}".format, subset=total_columns
    )


def get_styled_census_data(state: str) -> None:
    state_language_data = get_frame_for_state_vars(DetailedLanguageVars).loc[state]
    state_name = state_language_data.index[0]
    display(HTML(f"<h3 id='utilization'>WIC Utilization data for {state_name}</h3>"))
    display(get_wic_coverage_frame(state_name))
    display(
        HTML(f"<h3 id='state-lang'>Detailed language breakdowns for {state_name}</h3>")
    )
    detailed_language_data = get_percentages(state_language_data).T
    detailed_language_data.columns = ["percentage of speakers"]
    display(format_percentage_frame(detailed_language_data))
    county_data = get_county_census_data().loc[state]
    display(
        HTML(
            f"<h3 id='county-lang'>County-level language and poverty data for {state_name}</h3>"
        )
    )
    display(format_percentage_frame(county_data))


def get_styled_tribal_data(tribe: str) -> None:
    tribal_language_data = get_frame_for_tribal_areas(TotalPopulationVars).join(
        get_percentages(get_frame_for_tribal_areas(LanguageVars))
    )
    display(HTML(f"<h3 id='tribal'>Tribal language Data for {tribe}</h3>"))
    display(
        format_percentage_frame(
            (tribal_language_data)[tribal_language_data.index == tribe]
        )
    )
