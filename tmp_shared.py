from src.tvr_service.model import load_and_normalize, parse_robot
norm = load_and_normalize("Default.tvr2")
robot = parse_robot(norm, [5562, 5566])
shared = [node for node in robot.nodes.values() if node.shared]
print(len(shared))
diffs = set()
for node in shared:
    rel_long = node.relative_positions[5562]
    rel_short = node.relative_positions[5566]
    def offset(label):
        if label == "begin":
            return 0
        if "+" in label:
            return int(label.split("+")[1])
        if label.startswith("begin-"):
            return -int(label.split("-")[1])
        if label.startswith("begin") and len(label) > 5:
            return int(label[5:])
        raise ValueError(label)
    off_long = offset(rel_long)
    off_short = offset(rel_short)
    diffs.add(off_short - off_long)
print(diffs)
