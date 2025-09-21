from src.tvr_service.generator.robot_builder import generate_robot_segment

result = generate_robot_segment(
    'Default.tvr2',
    base_strokas=[5562, 5566],
    base_assignment={5562: 1, 5566: 5},
    sec0_value='MME55',
    output_tvr_path='generated_override.tvr2',
    duplicate_shared_filters=True,
    stroka_overrides={5566: 3},
)

print(result.stroka_mapping[5562], result.stroka_mapping[5566])
print(sorted(result.stroka_mapping.items())[:5])
