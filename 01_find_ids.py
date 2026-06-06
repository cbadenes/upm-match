"""
Script to find the `idPersonal` and `rfSignature` of all researchers in the UPM portal.
The results are saved in a CSV file and uploaded to MinIO for later use in the enrichment step.

The script sends a POST request to a PHP endpoint that returns an HTML page containing a JavaScript variable `dataSet` with the researchers' data.
It extracts this variable, parses it as JSON, and saves the relevant fields in a CSV file.
The CSV file contains the following columns:
- `idPersonal`: The unique identifier for each researcher.
- `rfSignature`: The researcher's signature, which is used to construct the URL for their profile page.
- `rfAdscription`: The researcher's affiliation.
- `rfRoll`: The researcher's role.

Use:
> uv run 01_find_ids.py
"""

import json
import re

import polars as pl
import requests
from loguru import logger

from utils.logging import setup_logging

from dbs import minio

URL = (
    "https://portalcientifico.upm.es/imarina_mod/templates/imarina/"
    "ipublic/get-ipublic-all-researchers-datatables.php"
)

PAYLOAD = {"idioma": "es"}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://portalcientifico.upm.es/",
}


def main():
    setup_logging("find_ids")
    minio.ensure_bucket(minio.RAW_BUCKET)
    logger.info("Sending POST request to the PHP endpoint...")
    response = requests.post(URL, data=PAYLOAD, headers=HEADERS, timeout=30)
    response.raise_for_status()

    match = re.search(r"var\s+dataSet\s*=\s*(\[.+?\]);", response.text, re.DOTALL)
    if not match:
        logger.error("`dataSet` was not found. First 500 response characters:")
        logger.error(response.text[:500])
        return

    dataset = json.loads(match.group(1))
    logger.info(f"Researchers found: {len(dataset)}")

    # List all available fields.
    # fields = list(dataset[0].keys())
    # print(f"Available fields ({len(fields)}):")
    # for field_name in fields:
    #     print(f"  - {field_name}")
    # print()

    # Preview the first 10 rows with all fields.
    df_full = pl.DataFrame(dataset, infer_schema_length=None)
    logger.info("Previewing the first 10 rows with all fields.")
    logger.info(
        df_full.drop(
            [
                "idStatus",
                "rfOfficialidType",
                "idOfficialidType",
                "idLocalPk",
                "dtLocalPk",
                "rfName",
                "rfFirstName",
                "rfSecondName",
                "idAuthorRol",
                "rfSignature2",
                "rfAdscriptionNameType1",
                "rfRoll",
                "rfPhoto",
                "rfPhoto2",
                "rfGender",
                "idGender",
                "rfPhotoExt",
                "rfPhoneExt",
                "rfAddres",
                "rfAddres2",
                "rfPostalCode",
                "rfCity",
                "rfProvince",
                "idProvince",
                "rfPhone",
                "rfNacionality",
                "idNacionality",
                "rfRegion",
                "idRegion",
                "rfCountry",
                "idCountry",
                "rfPhoneCode",
                "rfPhone2",
                "rfAncestorEntity",
            ]
        ).head(10)
    )
    logger.info("Preview complete.")

    # Save a CSV with idPersonal and rfSignature.
    df = df_full.select(
        [
            "idPersonal",
            "rfSignature",
            "rfAdscription",
            "rfRoll",
        ]
    ).sort("rfSignature")

    minio.upload_bytes(
        bucket_name=minio.RAW_BUCKET,
        data=df.write_csv().encode("utf-8"),
        object_name="researchers.csv",
        content_type="text/csv",
    )
    # output = Path(__file__).parent / "researchers.csv"
    # df.write_csv(output)
    # logger.info(f"Saved rows: {len(df)}")
    # logger.info(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
