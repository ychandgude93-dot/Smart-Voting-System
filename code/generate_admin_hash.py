from werkzeug.security import generate_password_hash
import sys

if len(sys.argv) < 2:
    print("Usage: python generate_admin_hash.py <password>")
    sys.exit(1)

password = sys.argv[1]
hash = generate_password_hash(password)
print(hash)
