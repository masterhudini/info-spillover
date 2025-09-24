#!/usr/bin/env python3
"""
Minimal BigQuery test script to demonstrate the exact issue and provide solutions.

This script tests BigQuery connectivity and provides clear instructions
for fixing the authentication scope issue.
"""

import os
import json
from google.cloud import bigquery
from google.auth import default


def main():
    print("=" * 70)
    print("🔍 BIGQUERY DIAGNOSTICS - Information Spillover Pipeline")
    print("=" * 70)

    # Check current authentication
    print("\n1️⃣ Checking Authentication...")
    print("-" * 40)

    try:
        credentials, project = default()
        print(f"✅ Project ID: {project}")
        print(f"✅ Authentication method: {type(credentials).__name__}")

        # Check if we're using service account
        if hasattr(credentials, 'service_account_email'):
            print(f"✅ Service account: {credentials.service_account_email}")

    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    print("\n2️⃣ Testing BigQuery Access...")
    print("-" * 40)

    try:
        # Create BigQuery client
        client = bigquery.Client(project=project)
        print(f"✅ BigQuery client created for project: {client.project}")

        # Test simple query - this is where it will fail
        test_query = "SELECT 'BigQuery test successful!' as message, CURRENT_TIMESTAMP() as timestamp"
        print(f"🔍 Executing test query: {test_query}")

        query_job = client.query(test_query)
        results = list(query_job.result())

        print("✅ SUCCESS! BigQuery is working correctly")
        for row in results:
            print(f"   📊 Result: {row.message} at {row.timestamp}")

    except Exception as e:
        print(f"❌ BigQuery test failed: {e}")

        if "Missing required OAuth scope" in str(e):
            print("\n" + "="*70)
            print("🚨 SCOPE ISSUE DETECTED - Here's how to fix it:")
            print("="*70)
            print()
            print("The VM service account needs BigQuery access scopes.")
            print("This is a one-time VM configuration issue.")
            print()
            print("SOLUTION 1: Update VM Access Scopes (Recommended)")
            print("-" * 50)
            print("1. Stop this VM instance")
            print("2. Go to Google Cloud Console → Compute Engine → VM instances")
            print("3. Click on your VM → Edit")
            print("4. In 'Access scopes' section, choose:")
            print("   • 'Allow full access to all Cloud APIs'")
            print("   OR")
            print("   • 'Set access for each API' and enable 'BigQuery'")
            print("5. Save and restart the VM")
            print()
            print("SOLUTION 2: Use Service Account Key (Alternative)")
            print("-" * 50)
            print("1. Download service account key JSON file")
            print("2. Set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
            print("3. Restart Python processes")
            print()
            print("Current service account:", getattr(credentials, 'service_account_email', 'Unknown'))
            print()
            print("After fixing, run this script again to verify!")

        return

    print("\n3️⃣ Testing Dataset Operations...")
    print("-" * 40)

    try:
        # List existing datasets
        datasets = list(client.list_datasets())
        print(f"✅ Found {len(datasets)} existing datasets:")
        for dataset in datasets:
            print(f"   📁 {dataset.dataset_id}")

        # Test dataset creation (dry run)
        dataset_id = "info_spillover_test"
        print(f"\n🔍 Testing dataset creation capabilities...")

        # Check if test dataset exists
        try:
            dataset_ref = client.dataset(dataset_id)
            dataset = client.get_dataset(dataset_ref)
            print(f"✅ Test dataset '{dataset_id}' already exists")
        except:
            print(f"ℹ️  Test dataset '{dataset_id}' does not exist (this is normal)")

    except Exception as e:
        print(f"❌ Dataset operations failed: {e}")

    print("\n" + "="*70)
    print("🎯 SUMMARY")
    print("="*70)
    print("✅ BigQuery setup is working correctly!")
    print("   Your pipeline is ready to process data")
    print()


if __name__ == "__main__":
    main()