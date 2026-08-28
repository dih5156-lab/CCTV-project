import zipfile

from scripts.datasets.inspect_aihub_kreid_dataset import inspect_label_zip


def _xml(
    *,
    person_id: str,
    upper_color: str,
    upper_defined: str,
    lower_color: str,
    lower_defined: str,
) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<xml>
  <FILE><name>{person_id}_frame.png</name></FILE>
  <OBJECT ID="{person_id}" TYPE="Human">
    <gender>male</gender>
    <upperclothes>long_sleeve</upperclothes>
    <upperclothes_color>{upper_color}</upperclothes_color>
    <defined_upperclothes_color>{upper_defined}</defined_upperclothes_color>
    <lowerclothes>long_pants</lowerclothes>
    <lowerclothes_color>{lower_color}</lowerclothes_color>
    <defined_lowerclothes_color>{lower_defined}</defined_lowerclothes_color>
  </OBJECT>
</xml>
"""


def test_inspect_label_zip_counts_only_defined_colors(tmp_path):
    label_zip = tmp_path / "labels.zip"
    with zipfile.ZipFile(label_zip, "w") as archive:
        archive.writestr(
            "H00001_1.xml",
            _xml(
                person_id="H00001",
                upper_color="white",
                upper_defined="true",
                lower_color="black",
                lower_defined="true",
            ),
        )
        archive.writestr(
            "H00002_1.xml",
            _xml(
                person_id="H00002",
                upper_color="red",
                upper_defined="false",
                lower_color="blue",
                lower_defined="true",
            ),
        )
        archive.writestr("broken.xml", "<xml>")

    result = inspect_label_zip(label_zip)

    assert result["xml_files"] == 3
    assert result["parsed_labels"] == 2
    assert result["malformed_xml"] == 1
    assert result["person_ids"] == 2
    assert result["upper_colors"] == {"white": 1}
    assert result["lower_colors"] == {"black": 1, "blue": 1}
    assert result["undefined_upper_colors"] == 1
    assert result["undefined_lower_colors"] == 0
    assert result["upper_clothes"] == {"long_sleeve": 2}
    assert result["lower_clothes"] == {"long_pants": 2}
