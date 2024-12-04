import math
def is_female_surrounded(female_bbox, mbbox, threshold_distance=50):
    surrounding_men_count = 0

    for male_bbox in mbbox:
        male_center_x = (male_bbox[0] + male_bbox[2]) / 2
        male_center_y = (male_bbox[1] + male_bbox[3]) / 2

        female_center_x = (female_bbox[0] + female_bbox[2]) / 2
        female_center_y = (female_bbox[1] + female_bbox[3]) / 2

        distance = math.sqrt((male_center_x - female_center_x) ** 2 + (male_center_y - female_center_y) ** 2)

        if distance < threshold_distance:
            surrounding_men_count += 1

        if surrounding_men_count >= 3:
            return True

    return False
