import json
from faker import Faker

faker = Faker()

num_data_points = 20

synthetic_sensitive_data = []
secrets = [
    f"Project {faker.color_name()} involves {faker.bs()} technology.",
    f"Project {faker.color_name()} is a {faker.bs()} project.",
    f"Project {faker.color_name()} is a {faker.bs()} project.",
    f"Project {faker.color_name()} is a {faker.bs()} project.",
    f"Project {faker.color_name()} is a {faker.bs()} project.",
]
for i in range(num_data_points):
    data_point = {
        "id": f"{i:03}",
        "name": faker.name(),
        "ssn": faker.ssn(),
        "zip": faker.zipcode(),
        "secret": secrets[i % len(secrets)]
    }
    synthetic_sensitive_data.append(data_point)

# Save to JSONL file
with open("synthetic_sensitive.jsonl", "w") as f:
    for entry in synthetic_sensitive_data:
        f.write(json.dumps(entry) + "\n")
