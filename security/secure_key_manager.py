"""
Secure Key Manager for Perp-DEX-Tools
Encrypts and manages API keys and private keys securely
"""

import os
import json
import getpass
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

logger = logging.getLogger(__name__)


class SecureKeyManager:
    """
    Manages encrypted storage of API keys and private keys

    Features:
    - Fernet encryption (AES-128)
    - Master password protection
    - Automatic .env migration
    - Secure key derivation (PBKDF2)
    """

    DEFAULT_ENCRYPTED_FILE = ".keys.enc"
    DEFAULT_SALT_FILE = ".salt"
    ITERATIONS = 480000  # OWASP recommended minimum

    def __init__(
        self,
        encrypted_file: str = None,
        salt_file: str = None,
        master_password: str = None
    ):
        """
        Initialize SecureKeyManager

        Args:
            encrypted_file: Path to encrypted keys file
            salt_file: Path to salt file
            master_password: Master password (will prompt if not provided)
        """
        self.encrypted_file = Path(encrypted_file or self.DEFAULT_ENCRYPTED_FILE)
        self.salt_file = Path(salt_file or self.DEFAULT_SALT_FILE)
        self._master_password = master_password
        self._fernet = None
        self._keys = {}

    def _get_master_password(self) -> str:
        """Get master password from user or stored value"""
        if self._master_password:
            return self._master_password

        # Prompt user for password
        password = getpass.getpass("Enter master password: ")
        if not password:
            raise ValueError("Master password cannot be empty")

        self._master_password = password
        return password

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2

        Args:
            password: Master password
            salt: Random salt bytes

        Returns:
            32-byte encryption key
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create new one"""
        if self.salt_file.exists():
            with open(self.salt_file, 'rb') as f:
                return f.read()

        # Generate new salt
        salt = os.urandom(16)
        with open(self.salt_file, 'wb') as f:
            f.write(salt)

        # Secure file permissions (Unix only)
        try:
            os.chmod(self.salt_file, 0o600)
        except Exception as e:
            logger.warning(f"Could not set salt file permissions: {e}")

        return salt

    def _get_fernet(self) -> Fernet:
        """Get or create Fernet cipher"""
        if self._fernet:
            return self._fernet

        password = self._get_master_password()
        salt = self._get_or_create_salt()
        key = self._derive_key(password, salt)
        self._fernet = Fernet(key)
        return self._fernet

    def initialize(self, force: bool = False) -> bool:
        """
        Initialize encrypted key storage

        Args:
            force: Force re-initialization even if file exists

        Returns:
            True if successful
        """
        if self.encrypted_file.exists() and not force:
            logger.info("Encrypted key file already exists")
            return True

        # Get password and confirm
        password = getpass.getpass("Create master password: ")
        confirm = getpass.getpass("Confirm master password: ")

        if password != confirm:
            raise ValueError("Passwords do not match")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        self._master_password = password

        # Create empty encrypted file
        self._keys = {}
        self.save_keys()

        logger.info("✅ Encrypted key storage initialized")
        return True

    def load_keys(self) -> Dict[str, Any]:
        """
        Load and decrypt keys from file

        Returns:
            Dictionary of decrypted keys
        """
        if not self.encrypted_file.exists():
            logger.warning("Encrypted key file does not exist. Run initialize() first.")
            return {}

        try:
            with open(self.encrypted_file, 'rb') as f:
                encrypted_data = f.read()

            fernet = self._get_fernet()
            decrypted_data = fernet.decrypt(encrypted_data)
            self._keys = json.loads(decrypted_data.decode())

            logger.info(f"✅ Loaded {len(self._keys)} keys from encrypted storage")
            return self._keys

        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            raise ValueError("Failed to decrypt keys. Check your master password.")

    def save_keys(self) -> bool:
        """
        Encrypt and save keys to file

        Returns:
            True if successful
        """
        try:
            fernet = self._get_fernet()
            json_data = json.dumps(self._keys, indent=2)
            encrypted_data = fernet.encrypt(json_data.encode())

            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)

            # Secure file permissions (Unix only)
            try:
                os.chmod(self.encrypted_file, 0o600)
            except Exception as e:
                logger.warning(f"Could not set encrypted file permissions: {e}")

            logger.info(f"✅ Saved {len(self._keys)} keys to encrypted storage")
            return True

        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
            return False

    def set_key(self, key_name: str, key_value: str) -> bool:
        """
        Set a key value

        Args:
            key_name: Name of the key
            key_value: Value to store

        Returns:
            True if successful
        """
        if not key_value:
            logger.warning(f"Skipping empty key: {key_name}")
            return False

        self._keys[key_name] = key_value
        return self.save_keys()

    def get_key(self, key_name: str, default: str = None) -> Optional[str]:
        """
        Get a key value

        Args:
            key_name: Name of the key
            default: Default value if key not found

        Returns:
            Key value or default
        """
        return self._keys.get(key_name, default)

    def delete_key(self, key_name: str) -> bool:
        """
        Delete a key

        Args:
            key_name: Name of the key to delete

        Returns:
            True if key was deleted
        """
        if key_name in self._keys:
            del self._keys[key_name]
            self.save_keys()
            logger.info(f"✅ Deleted key: {key_name}")
            return True

        logger.warning(f"Key not found: {key_name}")
        return False

    def list_keys(self) -> list:
        """
        List all key names (not values)

        Returns:
            List of key names
        """
        return list(self._keys.keys())

    def migrate_from_env(self, env_file: str = ".env", backup: bool = True) -> bool:
        """
        Migrate keys from .env file to encrypted storage

        Args:
            env_file: Path to .env file
            backup: Create backup of .env file

        Returns:
            True if successful
        """
        env_path = Path(env_file)

        if not env_path.exists():
            logger.warning(f".env file not found: {env_file}")
            return False

        # Backup .env file
        if backup:
            backup_path = env_path.with_suffix('.env.backup')
            import shutil
            shutil.copy2(env_path, backup_path)
            logger.info(f"✅ Backed up .env to {backup_path}")

        # Read .env file
        migrated_count = 0
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if value:
                        self._keys[key] = value
                        migrated_count += 1

        # Save to encrypted storage
        self.save_keys()

        logger.info(f"✅ Migrated {migrated_count} keys from {env_file}")

        # Optionally remove sensitive data from .env
        print("\n⚠️  Migration complete!")
        print(f"   Migrated {migrated_count} keys to encrypted storage")
        print(f"   Backup saved to: {env_path.with_suffix('.env.backup')}")
        print("\nNext steps:")
        print("1. Verify encrypted keys work: python -m security.secure_key_manager list")
        print(f"2. Remove sensitive data from {env_file}")
        print("3. Update your code to use SecureKeyManager.get_key()")

        return True

    def export_to_env(self, output_file: str = ".env.decrypted") -> bool:
        """
        Export keys to .env format (for debugging/testing only)

        Args:
            output_file: Output file path

        Returns:
            True if successful
        """
        if not self._keys:
            logger.warning("No keys loaded. Call load_keys() first.")
            return False

        output_path = Path(output_file)

        with open(output_path, 'w') as f:
            f.write("# Exported from encrypted storage\n")
            f.write("# WARNING: This file contains sensitive data\n\n")

            for key, value in sorted(self._keys.items()):
                f.write(f'{key}="{value}"\n')

        # Secure file permissions
        try:
            os.chmod(output_path, 0o600)
        except Exception:
            pass

        logger.info(f"✅ Exported {len(self._keys)} keys to {output_file}")
        print(f"\n⚠️  WARNING: {output_file} contains UNENCRYPTED secrets!")
        print("   Delete this file after use.")

        return True


def main():
    """CLI interface for SecureKeyManager"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m security.secure_key_manager <command> [args]")
        print("\nCommands:")
        print("  init                  - Initialize encrypted key storage")
        print("  migrate [.env]        - Migrate keys from .env file")
        print("  set <key> <value>     - Set a key")
        print("  get <key>             - Get a key value")
        print("  delete <key>          - Delete a key")
        print("  list                  - List all key names")
        print("  export [output.env]   - Export to .env format (DEBUG ONLY)")
        return

    command = sys.argv[1]
    manager = SecureKeyManager()

    try:
        if command == "init":
            manager.initialize()
            print("✅ Initialized encrypted key storage")

        elif command == "migrate":
            env_file = sys.argv[2] if len(sys.argv) > 2 else ".env"
            manager.initialize(force=False)
            manager.migrate_from_env(env_file)

        elif command == "set":
            if len(sys.argv) < 4:
                print("Usage: set <key> <value>")
                return

            manager.load_keys()
            key_name = sys.argv[2]
            key_value = sys.argv[3]
            manager.set_key(key_name, key_value)
            print(f"✅ Set key: {key_name}")

        elif command == "get":
            if len(sys.argv) < 3:
                print("Usage: get <key>")
                return

            manager.load_keys()
            key_name = sys.argv[2]
            value = manager.get_key(key_name)

            if value:
                print(f"{key_name}={value}")
            else:
                print(f"❌ Key not found: {key_name}")

        elif command == "delete":
            if len(sys.argv) < 3:
                print("Usage: delete <key>")
                return

            manager.load_keys()
            key_name = sys.argv[2]
            manager.delete_key(key_name)

        elif command == "list":
            manager.load_keys()
            keys = manager.list_keys()
            print(f"\n📋 Stored keys ({len(keys)}):")
            for key in sorted(keys):
                print(f"  - {key}")

        elif command == "export":
            output_file = sys.argv[2] if len(sys.argv) > 2 else ".env.decrypted"
            manager.load_keys()
            manager.export_to_env(output_file)

        else:
            print(f"❌ Unknown command: {command}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
