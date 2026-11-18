
from src.video_analytics.zones import RestrictedZone


def test_point_inside_zone():
    zone = RestrictedZone(
        name="test",
        normalized_points=[
            [0.5, 0.5],
            [0.9, 0.5],
            [0.9, 0.9],
            [0.5, 0.9],
        ],
    )

    assert zone.contains((70, 70), 100, 100)
    assert not zone.contains((20, 20), 100, 100)


def test_entry_events_not_repeated():
    zone = RestrictedZone(
        name="test",
        normalized_points=[
            [0.5, 0.5],
            [0.9, 0.5],
            [0.9, 0.9],
        ],
    )

    assert zone.update({1}) == {1}
    assert zone.update({1}) == set()
    assert zone.update({1, 2}) == {2}

    zone.update(set())
    assert zone.update({1}) == {1}
