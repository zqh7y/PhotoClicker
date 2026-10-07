import cv2
import numpy as np
import pytest

from autoclicker import Finder, find_matches, load_template, parse_region, parse_scales


def make_scene():
    """Noisy background with two copies of a distinctive patch."""
    rng = np.random.default_rng(0)
    screen = rng.integers(0, 60, (400, 600, 3), dtype=np.uint8)
    patch = np.zeros((30, 40, 3), np.uint8)
    cv2.rectangle(patch, (2, 2), (37, 27), (0, 200, 255), -1)
    cv2.circle(patch, (20, 15), 8, (255, 255, 255), -1)
    cv2.putText(patch, "E", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    screen[100:130, 200:240] = patch
    screen[300:330, 500:540] = patch
    return screen, patch


def test_finds_center_of_best_match():
    screen, patch = make_scene()
    matches = find_matches(screen, patch, 0.9)
    assert len(matches) == 1
    x, y, score = matches[0]
    assert (x, y) in [(220, 115), (520, 315)]
    assert score > 0.99


def test_find_all_returns_each_copy_once():
    screen, patch = make_scene()
    matches = find_matches(screen, patch, 0.9, find_all=True)
    assert sorted((x, y) for x, y, _ in matches) == [(220, 115), (520, 315)]


def test_nothing_found_below_confidence():
    screen, _ = make_scene()
    other = np.full((30, 40, 3), (200, 50, 50), np.uint8)
    cv2.line(other, (0, 0), (39, 29), (0, 255, 0), 3)
    assert find_matches(screen, other, 0.85) == []


def test_scales_find_a_resized_target():
    screen, patch = make_scene()
    big = cv2.resize(patch, None, fx=1.25, fy=1.25)
    screen[20:20 + big.shape[0], 20:20 + big.shape[1]] = big
    small_screen = screen[:90, :120].copy()
    assert find_matches(small_screen, patch, 0.85) == []
    matches = find_matches(small_screen, patch, 0.85, scales=(1.0, 1.25))
    assert len(matches) == 1
    x, y, _ = matches[0]
    assert abs(x - (20 + big.shape[1] // 2)) <= 1
    assert abs(y - (20 + big.shape[0] // 2)) <= 1


def test_grayscale_and_transparent_png(tmp_path):
    screen, patch = make_scene()
    rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    rgba[:5, :, 3] = 0  # transparent top strip
    path = tmp_path / "button.png"
    cv2.imwrite(str(path), rgba)

    tpl, mask = load_template(path)
    assert mask is not None
    matches = find_matches(screen, tpl, 0.9, mask=mask, find_all=True)
    assert sorted((x, y) for x, y, _ in matches) == [(220, 115), (520, 315)]

    gray_tpl, gray_mask = load_template(path, grayscale=True)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    matches = find_matches(gray_screen, gray_tpl, 0.9, mask=gray_mask, find_all=True)
    assert len(matches) == 2


def test_template_bigger_than_screen_is_skipped():
    screen, patch = make_scene()
    assert find_matches(screen[:20, :20], patch, 0.5) == []


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_template(tmp_path / "nope.png")


def test_parsers():
    assert parse_scales("0.8, 1,1.25") == (0.8, 1.0, 1.25)
    assert parse_region("10,20,300,400") == {"left": 10, "top": 20, "width": 300, "height": 400}


def make_big_scene():
    """A full-HD-ish screen with two copies of a 120x90 button."""
    screen, patch = make_scene()
    big_patch = cv2.resize(patch, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    screen = cv2.resize(screen, (1600, 900), interpolation=cv2.INTER_NEAREST)
    screen[101:191, 333:453] = big_patch
    screen[600:690, 1201:1321] = big_patch
    return screen, big_patch


def test_big_template_uses_fast_search_and_exact_center():
    screen, big = make_big_scene()
    matches = find_matches(screen, big, 0.9)
    assert len(matches) == 1
    assert matches[0][:2] in [(393, 146), (1261, 645)]
    assert matches[0][2] > 0.99


def test_big_template_find_all_and_sizes():
    screen, big = make_big_scene()
    matches = find_matches(screen, big, 0.9, find_all=True)
    assert sorted((x, y) for x, y, _ in matches) == [(393, 146), (1261, 645)]

    smaller = cv2.resize(big, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    matches = find_matches(screen, smaller, 0.85, scales=(0.9, 1.0, 1.1, 1.25), find_all=True)
    assert len(matches) == 2
    for x, y, _ in matches:
        assert min(abs(x - 393) + abs(y - 146), abs(x - 1261) + abs(y - 645)) <= 3


def test_shape_mode_finds_recolored_and_darker_copies():
    screen, patch = make_scene()
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 70) % 180
    screen[300:330, 500:540] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # other color
    screen[200:230, 50:90] = (patch * 0.6).astype(np.uint8)  # darker

    by_color = find_matches(screen, patch, 0.9, find_all=True)
    assert (520, 315) not in [(x, y) for x, y, _ in by_color]

    by_shape = find_matches(screen, patch, 0.8, find_all=True, shape=True)
    assert sorted((x, y) for x, y, _ in by_shape) == [(70, 215), (220, 115), (520, 315)]


def test_shape_mode_ignores_plain_areas():
    flat = np.full((300, 300, 3), 90, np.uint8)
    _, patch = make_scene()
    assert find_matches(flat, patch, 0.5, find_all=True, shape=True) == []


def test_finder_matches_find_matches():
    screen, big = make_big_scene()
    finder = Finder(big, scales=(0.9, 1.0, 1.1))
    best = finder.find(screen, 0.9)
    assert len(best) == 1 and best[0][:2] in [(393, 146), (1261, 645)]
    found = finder.find(screen, 0.9, find_all=True)
    assert sorted((x, y) for x, y, _ in found) == [(393, 146), (1261, 645)]


def test_finder_any_color_and_size():
    screen, big = make_big_scene()
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 70) % 180
    other = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), None, fx=1.1, fy=1.1)
    screen[101:191, 333:453] = screen[0:90, 0:120]  # remove the first copy
    screen[300:300 + other.shape[0], 700:700 + other.shape[1]] = other

    finder = Finder(big, scales=(0.9, 1.0, 1.1), shape=True)
    found = finder.find(screen, 0.75, find_all=True)
    centers = sorted((round(x), round(y)) for x, y, _ in found)
    assert len(centers) == 2
    assert abs(centers[0][0] - (700 + other.shape[1] // 2)) <= 3
    assert abs(centers[1][0] - 1261) <= 3


def test_finder_ignores_plain_screen():
    _, big = make_big_scene()
    flat = np.full((600, 800, 3), 90, np.uint8)
    assert Finder(big, shape=True).find(flat, 0.5, find_all=True) == []
