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
    display(HTML(f"<h2 id='utilization'>WIC Utilization data for {state_name}</h2>"))
    display(get_wic_coverage_frame(state_name))
    display(
        HTML(f"<h2 id='state-lang'>Detailed language breakdowns for {state_name}</h2>")
    )
    detailed_language_data = get_percentages(state_language_data).T
    detailed_language_data.columns = ["percentage of speakers"]
    display(format_percentage_frame(detailed_language_data))
    county_data = get_county_census_data().loc[state]
    display(
        HTML(
            f"<h2 id='county-lang'>County-level language and poverty data for {state_name}</h2>"
        )
    )
    display(format_percentage_frame(county_data))
