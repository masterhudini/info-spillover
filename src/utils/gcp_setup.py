"""
Google Cloud Platform Setup Utilities
Helper functions for configuring GCP authentication and services
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict
from google.cloud import bigquery
from google.oauth2 import service_account
from google.auth import default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPAuthenticator:
    """Helper class for Google Cloud authentication setup"""

    @staticmethod
    def check_credentials() -> Dict[str, str]:
        """
        Check current Google Cloud authentication status

        Returns:
            Dict with authentication information
        """
        auth_info = {
            'method': None,
            'project_id': None,
            'status': 'unknown'
        }

        try:
            # Try to get default credentials
            credentials, project = default()

            auth_info['project_id'] = project
            auth_info['status'] = 'authenticated'

            # Determine authentication method
            if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                auth_info['method'] = 'service_account_key'
                auth_info['key_path'] = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                logger.info(f"✅ Using service account key: {auth_info['key_path']}")
            else:
                auth_info['method'] = 'default_application'
                logger.info("✅ Using Application Default Credentials")

            logger.info(f"✅ Authenticated for project: {project}")

        except Exception as e:
            auth_info['status'] = 'failed'
            auth_info['error'] = str(e)
            logger.error(f"❌ Authentication failed: {str(e)}")

        return auth_info

    @staticmethod
    def setup_service_account_auth(key_path: str) -> bool:
        """
        Setup authentication using service account key file

        Args:
            key_path: Path to service account JSON key file

        Returns:
            True if successful, False otherwise
        """
        try:
            key_path = Path(key_path)

            if not key_path.exists():
                logger.error(f"❌ Service account key file not found: {key_path}")
                return False

            # Validate JSON structure
            with open(key_path, 'r') as f:
                key_data = json.load(f)

            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in key_data]

            if missing_fields:
                logger.error(f"❌ Invalid service account key - missing fields: {missing_fields}")
                return False

            # Set environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(key_path.absolute())

            logger.info(f"✅ Service account authentication configured: {key_path}")
            logger.info(f"✅ Project ID: {key_data.get('project_id')}")
            logger.info(f"✅ Service Account: {key_data.get('client_email')}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup service account auth: {str(e)}")
            return False

    @staticmethod
    def setup_user_auth() -> bool:
        """
        Setup authentication using user credentials (gcloud auth)

        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess

            logger.info("Setting up user authentication...")
            logger.info("This will open a web browser for authentication")

            # Run gcloud auth application-default login
            result = subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("✅ User authentication setup successful")
                return True
            else:
                logger.error(f"❌ gcloud auth failed: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error("❌ gcloud CLI not found. Please install Google Cloud SDK")
            logger.error("Visit: https://cloud.google.com/sdk/docs/install")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to setup user auth: {str(e)}")
            return False

    @staticmethod
    def test_bigquery_access(project_id: Optional[str] = None) -> bool:
        """
        Test BigQuery access with current credentials

        Args:
            project_id: GCP project ID (optional)

        Returns:
            True if access is successful, False otherwise
        """
        try:
            # Initialize BigQuery client
            client = bigquery.Client(project=project_id)

            # Test basic query
            test_query = "SELECT 1 as test, CURRENT_TIMESTAMP() as timestamp"
            query_job = client.query(test_query)
            results = list(query_job.result())

            logger.info(f"✅ BigQuery access test successful")
            logger.info(f"✅ Project: {client.project}")
            logger.info(f"✅ Test result: {results[0]}")

            # List datasets
            datasets = list(client.list_datasets())
            logger.info(f"✅ Can access {len(datasets)} datasets")

            return True

        except Exception as e:
            logger.error(f"❌ BigQuery access test failed: {str(e)}")
            return False

    @staticmethod
    def required_permissions() -> list:
        """
        Return list of required BigQuery permissions for this project

        Returns:
            List of IAM permissions
        """
        return [
            "bigquery.datasets.create",
            "bigquery.datasets.get",
            "bigquery.tables.create",
            "bigquery.tables.get",
            "bigquery.tables.getData",
            "bigquery.tables.updateData",
            "bigquery.jobs.create",
            "bigquery.jobs.get"
        ]

    @staticmethod
    def check_permissions(project_id: Optional[str] = None) -> Dict:
        """
        Check if current credentials have required permissions

        Args:
            project_id: GCP project ID (optional)

        Returns:
            Dict with permission check results
        """
        try:
            from google.cloud import resourcemanager

            client = bigquery.Client(project=project_id)

            # Get current project
            project_id = project_id or client.project

            logger.info(f"Checking BigQuery permissions for project: {project_id}")

            # Try to perform operations that require specific permissions
            permission_tests = {
                'list_datasets': False,
                'create_dataset': False,
                'run_query': False,
                'create_table': False
            }

            # Test listing datasets
            try:
                list(client.list_datasets())
                permission_tests['list_datasets'] = True
                logger.info("✅ Can list datasets")
            except Exception as e:
                logger.warning(f"❌ Cannot list datasets: {str(e)}")

            # Test running queries
            try:
                test_query = "SELECT 1"
                query_job = client.query(test_query)
                list(query_job.result())
                permission_tests['run_query'] = True
                logger.info("✅ Can run queries")
            except Exception as e:
                logger.warning(f"❌ Cannot run queries: {str(e)}")

            return {
                'project_id': project_id,
                'permissions': permission_tests,
                'all_permissions': all(permission_tests.values())
            }

        except Exception as e:
            logger.error(f"❌ Permission check failed: {str(e)}")
            return {'error': str(e)}


def quick_setup_guide():
    """Print quick setup guide for Google Cloud authentication"""

    print("\n" + "="*60)
    print("🔧 GOOGLE CLOUD SETUP GUIDE")
    print("="*60)

    print("\n📋 OPTION 1: Service Account (Recommended for production)")
    print("1. Go to Google Cloud Console → IAM & Admin → Service Accounts")
    print("2. Create a new service account with BigQuery permissions")
    print("3. Download the JSON key file")
    print("4. Set environment variable:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")

    print("\n📋 OPTION 2: User Authentication (For development)")
    print("1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
    print("2. Run: gcloud auth application-default login")
    print("3. Follow browser authentication flow")

    print("\n🔑 REQUIRED BIGQUERY PERMISSIONS:")
    for permission in GCPAuthenticator.required_permissions():
        print(f"   • {permission}")

    print("\n✅ VERIFY SETUP:")
    print("Run: python -c \"from src.utils.gcp_setup import GCPAuthenticator; GCPAuthenticator.test_bigquery_access()\"")
    print("="*60)


def main():
    """Main function for interactive setup"""

    print("🚀 Google Cloud Platform Authentication Setup")
    print("="*50)

    # Check current status
    auth_info = GCPAuthenticator.check_credentials()

    if auth_info['status'] == 'authenticated':
        print(f"✅ Already authenticated!")
        print(f"   Method: {auth_info['method']}")
        print(f"   Project: {auth_info['project_id']}")

        # Test BigQuery access
        if GCPAuthenticator.test_bigquery_access():
            print("✅ BigQuery access confirmed - you're ready to go!")
        else:
            print("❌ BigQuery access failed - check permissions")

    else:
        print("❌ Not authenticated")
        quick_setup_guide()

        # Interactive setup
        choice = input("\nChoose setup method (1=Service Account, 2=User Auth, 3=Manual): ")

        if choice == "1":
            key_path = input("Enter path to service account JSON key: ")
            if GCPAuthenticator.setup_service_account_auth(key_path):
                GCPAuthenticator.test_bigquery_access()
        elif choice == "2":
            GCPAuthenticator.setup_user_auth()
        else:
            print("Please follow the manual setup guide above")


if __name__ == "__main__":
    main()