#!/usr/bin/env python3
"""
Comprehensive BigQuery test for information spillover project.
Tests all operations needed for the pipeline.
"""

import os
import sys
sys.path.append('/home/Hudini/projects/info_spillover')

from src.data.bigquery_client import BigQueryClient
from src.utils.gcp_setup import GCPAuthenticator


def test_basic_connection():
    """Test basic BigQuery connectivity"""
    print("\n🔍 Testing Basic Connection...")
    print("-" * 50)

    try:
        auth_info = GCPAuthenticator.check_credentials()
        if auth_info['status'] != 'authenticated':
            print("❌ Authentication failed")
            return False

        print(f"✅ Project: {auth_info['project_id']}")
        print(f"✅ Method: {auth_info['method']}")

        # Test BigQuery access
        if GCPAuthenticator.test_bigquery_access():
            print("✅ BigQuery connection successful")
            return True
        else:
            print("❌ BigQuery connection failed")
            return False

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_dataset_operations():
    """Test dataset creation and management"""
    print("\n🔍 Testing Dataset Operations...")
    print("-" * 50)

    try:
        # Initialize client with test dataset
        client = BigQueryClient(dataset_id="info_spillover_test")
        print("✅ BigQuery client initialized")
        print(f"✅ Test dataset created/verified: info_spillover_test")
        return True

    except Exception as e:
        print(f"❌ Dataset operations failed: {e}")
        return False


def test_table_creation():
    """Test table creation for the pipeline"""
    print("\n🔍 Testing Table Creation...")
    print("-" * 50)

    try:
        client = BigQueryClient(dataset_id="info_spillover_test")

        # Test posts table creation
        posts_table = client.create_posts_table()
        print("✅ Posts/comments table created/verified")

        # Test prices table creation
        prices_table = client.create_prices_table()
        print("✅ Crypto prices table created/verified")

        return True

    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        return False


def test_query_operations():
    """Test querying capabilities"""
    print("\n🔍 Testing Query Operations...")
    print("-" * 50)

    try:
        client = BigQueryClient(dataset_id="info_spillover_test")

        # Test simple query
        test_query = """
        SELECT
            'BigQuery' as service,
            CURRENT_TIMESTAMP() as test_time,
            'Working correctly!' as status
        """

        df = client.query_data(test_query)
        print(f"✅ Query executed successfully, returned {len(df)} rows")
        print(f"✅ Result: {df.iloc[0]['status']}")

        return True

    except Exception as e:
        print(f"❌ Query operations failed: {e}")
        return False


def cleanup_test_resources():
    """Clean up test dataset"""
    print("\n🧹 Cleaning up test resources...")
    print("-" * 50)

    try:
        from google.cloud import bigquery
        client = bigquery.Client()

        # Delete test dataset
        dataset_ref = client.dataset("info_spillover_test")
        client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
        print("✅ Test dataset cleaned up")

    except Exception as e:
        print(f"⚠️  Cleanup warning (not critical): {e}")


def main():
    """Run comprehensive BigQuery tests"""
    print("=" * 70)
    print("🧪 COMPREHENSIVE BIGQUERY TEST")
    print("Information Spillover Project")
    print("=" * 70)

    tests = [
        ("Basic Connection", test_basic_connection),
        ("Dataset Operations", test_dataset_operations),
        ("Table Creation", test_table_creation),
        ("Query Operations", test_query_operations),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Cleanup
    cleanup_test_resources()

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("BigQuery is fully configured for your information spillover pipeline.")
        print("You can now run: python src/main_pipeline.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above and fix BigQuery configuration.")
    print("=" * 70)


if __name__ == "__main__":
    main()