import json
import sys
from pathlib import Path

try:
    import jsonschema
except Exception:
    print('jsonschema not available', file=sys.stderr)
    raise


def validate(parsed_path: Path, schema_path: Path) -> int:
    with parsed_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    with schema_path.open('r', encoding='utf-8') as f:
        schema = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(data))
    if not errors:
        print('OK: parsed document conforms to schema')
        return 0

    print(f'Found {len(errors)} schema validation errors:')
    for err in errors:
        # print a concise description
        path = ''.join([f'/{p}' for p in err.path])
        print(f'- {err.message} (at {path})')
    return 2


def main():
    if len(sys.argv) < 3:
        print('Usage: validate_parsed.py <parsed.json> <schema.json>', file=sys.stderr)
        sys.exit(2)
    parsed = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    if not parsed.exists():
        print('Parsed file not found:', parsed, file=sys.stderr)
        sys.exit(2)
    if not schema.exists():
        print('Schema file not found:', schema, file=sys.stderr)
        sys.exit(2)
    code = validate(parsed, schema)
    sys.exit(code)


if __name__ == '__main__':
    main()
