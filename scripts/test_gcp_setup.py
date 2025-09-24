#!/usr/bin/env python3
"""
Test script for Google Cloud Platform setup
Quick validation of BigQuery authentication and permissions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.gcp_setup import GCPAuthenticator, quick_setup_guide
from src.data.bigquery_client import BigQueryClient
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main test function"""

    print("\n" + "="*60)
    print("🧪 TESTING GOOGLE CLOUD PLATFORM SETUP")
    print("="*60)

    try:
        print("\n1️⃣ Checking Google Cloud Authentication...")
        print("-" * 40)

        # Check authentication status
        auth_info = GCPAuthenticator.check_credentials()

        if auth_info['status'] != 'authenticated':
            print("❌ Authentication failed!")
            print("\n🔧 Setup required:")
            quick_setup_guide()
            return False

        print("\n2️⃣ Testing BigQuery Access...")
        print("-" * 40)

        # Test BigQuery access
        if not GCPAuthenticator.test_bigquery_access(auth_info.get('project_id')):
            print("❌ BigQuery access test failed!")
            return False

        print("\n3️⃣ Testing BigQuery Client...")
        print("-" * 40)

        # Test BigQuery client initialization
        try:
            bq_client = BigQueryClient(project_id=auth_info.get('project_id'))
            print("✅ BigQuery client initialized successfully!")
            print(f"   Project: {bq_client.project_id}")
            print(f"   Dataset: {bq_client.dataset_id}")
        except Exception as e:
            print(f"❌ BigQuery client initialization failed: {str(e)}")
            return False

        print("\n4️⃣ Testing Data Loading Capabilities...")
        print("-" * 40)

        # Test basic query capability
        try:
            test_query = """
            SELECT
                'test' as source,
                CURRENT_TIMESTAMP() as timestamp,
                1234 as test_number
            """
            result = bq_client.query_data(test_query)

            if not result.empty:
                print("✅ BigQuery data querying works!")
                print(f"   Test result: {result.iloc[0].to_dict()}")
            else:
                print("⚠️ Query returned empty result")

        except Exception as e:
            print(f"❌ BigQuery querying failed: {str(e)}")
            return False

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Google Cloud Platform is properly configured")
        print("✅ BigQuery authentication successful")
        print("✅ Data pipeline is ready to run")
        print("\n🚀 You can now run the main pipeline:")
        print("   python src/main_pipeline.py")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure Google Cloud SDK is installed")
        print("2. Check your authentication method:")
        print("   • Service Account: GOOGLE_APPLICATION_CREDENTIALS env var")
        print("   • User Auth: gcloud auth application-default login")
        print("3. Verify BigQuery permissions in Google Cloud Console")
        print("\nFor detailed setup guide, run:")
        print("   python src/utils/gcp_setup.py")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)