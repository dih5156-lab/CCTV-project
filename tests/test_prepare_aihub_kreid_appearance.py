import zipfile

from scripts.datasets.prepare_aihub_kreid_appearance import (
    _select_evenly,
    map_color,
    prepare_split,
)


def _xml(person_id: str, image_name: str, upper: str, lower: str) -> str:
    return f"""<xml><FILE><name>{image_name}</name></FILE>
<OBJECT ID="{person_id}" TYPE="Human">
<gender>female</gender><upperclothes>long_sleeve</upperclothes>
<upperclothes_color>{upper}</upperclothes_color>
<defined_upperclothes_color>true</defined_upperclothes_color>
<lowerclothes>long_pants</lowerclothes><lowerclothes_color>{lower}</lowerclothes_color>
<defined_lowerclothes_color>true</defined_lowerclothes_color>
</OBJECT></xml>"""


def test_map_color_normalizes_aihub_values():
    assert map_color("navy") == "navy"
    assert map_color("흰색") == "white"
    assert map_color("orange") == "orange"
    assert map_color("black") == "black"


def test_select_evenly_preserves_rare_color_pairs_within_limit():
    rows = [
        {"image_name": f"{index}.png", "upper_color": "black", "lower_color": "blue"}
        for index in range(5)
    ]
    rows[2]["lower_color"] = "orange"

    selected = _select_evenly(rows, limit=2)

    assert len(selected) == 2
    assert any(row["lower_color"] == "orange" for row in selected)


def test_prepare_split_limits_each_person_and_extracts_matching_images(tmp_path):
    labels = tmp_path / "labels.zip"
    sources = tmp_path / "sources.zip"
    with zipfile.ZipFile(labels, "w") as label_archive, zipfile.ZipFile(sources, "w") as source_archive:
        for person_id in ("H00001", "H00002"):
            for index in range(4):
                image_name = f"{person_id}_{index}.png"
                label_archive.writestr(
                    image_name.replace(".png", ".xml"),
                    _xml(person_id, image_name, "navy", "흰색"),
                )
                source_archive.writestr(image_name, f"image-{person_id}-{index}".encode())

    output_dir = tmp_path / "output"
    rows, summary = prepare_split(
        label_zip=labels,
        source_zip=sources,
        output_dir=output_dir,
        split="train",
        max_images_per_person=2,
    )

    assert len(rows) == 4
    assert summary["selected_images"] == 4
    assert summary["person_ids"] == 2
    assert {row["upper_color"] for row in rows} == {"navy"}
    assert {row["lower_color"] for row in rows} == {"white"}
    assert {row["split"] for row in rows} == {"train"}
    assert all((output_dir / row["image_path"]).is_file() for row in rows)


def test_prepare_split_reads_bag_and_hat_items(tmp_path):
    labels = tmp_path / "labels.zip"
    sources = tmp_path / "sources.zip"
    xml = _xml("H00001", "H00001.png", "black", "blue").replace(
        "</xml>",
        '<ITEM_LIST><ITEM TYPE="IE"><kind_of>bag</kind_of></ITEM><ITEM TYPE="IA"><kind_of>hat</kind_of></ITEM></ITEM_LIST></xml>',
    )
    with zipfile.ZipFile(labels, "w") as label_archive, zipfile.ZipFile(sources, "w") as source_archive:
        label_archive.writestr("H00001.xml", xml)
        source_archive.writestr("H00001.png", b"image")

    rows, _ = prepare_split(
        label_zip=labels,
        source_zip=sources,
        output_dir=tmp_path / "output",
        split="train",
        max_images_per_person=1,
    )

    assert rows[0]["bag"] == "yes"
    assert rows[0]["hat"] == "yes"
