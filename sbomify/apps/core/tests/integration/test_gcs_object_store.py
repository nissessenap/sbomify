"""
Integration tests for GCSObjectStoreClient against fake-gcs-server.

These tests require the fake-gcs-server Docker service to be running.
They are automatically skipped when GCS_ENDPOINT_URL is not set or not reachable.
"""

import os
import uuid

import pytest
from django.test import override_settings

from sbomify.apps.core.object_store import GCSObjectStoreClient, StorageClient

# Skip entire module if fake-gcs-server is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("GCS_ENDPOINT_URL"),
    reason="GCS integration tests require fake-gcs-server (run via docker-compose.tests.yml)",
)

TEST_BUCKET = "integration-test-bucket"


@pytest.fixture(scope="module")
def gcs_client():
    """Create a GCSObjectStoreClient pointing at fake-gcs-server."""
    endpoint = os.environ["GCS_ENDPOINT_URL"]
    project = os.environ.get("GCS_PROJECT_ID", "test-project")
    client = GCSObjectStoreClient(
        project_id=project,
        endpoint_url=endpoint,
        use_emulator=True,
    )
    # Create test bucket (fake-gcs-server auto-creates on write, but be explicit)
    from google.api_core.exceptions import Conflict

    try:
        client._client.create_bucket(TEST_BUCKET)
    except Conflict:
        pass  # Bucket already exists
    return client


@pytest.fixture
def unique_key():
    """Generate a unique key for each test to avoid cross-test interference."""
    return f"test/{uuid.uuid4().hex}"


class TestGCSIntegration:
    def test_put_and_get_object(self, gcs_client, unique_key):
        data = b"hello from GCS integration test"
        gcs_client.put_object(TEST_BUCKET, unique_key, data)
        result = gcs_client.get_object(TEST_BUCKET, unique_key)
        assert result == data

    def test_get_object_missing_key_returns_none(self, gcs_client, unique_key):
        result = gcs_client.get_object(TEST_BUCKET, unique_key)
        assert result is None

    def test_object_exists_true(self, gcs_client, unique_key):
        gcs_client.put_object(TEST_BUCKET, unique_key, b"exists")
        assert gcs_client.object_exists(TEST_BUCKET, unique_key) is True

    def test_object_exists_false(self, gcs_client, unique_key):
        assert gcs_client.object_exists(TEST_BUCKET, unique_key) is False

    def test_delete_object(self, gcs_client, unique_key):
        gcs_client.put_object(TEST_BUCKET, unique_key, b"to-delete")
        gcs_client.delete_object(TEST_BUCKET, unique_key)
        assert gcs_client.get_object(TEST_BUCKET, unique_key) is None

    def test_upload_and_download_file(self, gcs_client, unique_key, tmp_path):
        src = tmp_path / "upload.txt"
        dst = tmp_path / "download.txt"
        src.write_bytes(b"file round-trip test")

        gcs_client.upload_file(TEST_BUCKET, str(src), unique_key)
        gcs_client.download_file(TEST_BUCKET, unique_key, str(dst))

        assert dst.read_bytes() == b"file round-trip test"

    def test_overwrite_object(self, gcs_client, unique_key):
        gcs_client.put_object(TEST_BUCKET, unique_key, b"version-1")
        gcs_client.put_object(TEST_BUCKET, unique_key, b"version-2")
        assert gcs_client.get_object(TEST_BUCKET, unique_key) == b"version-2"

    def test_generate_presigned_url(self, gcs_client, unique_key):
        gcs_client.put_object(TEST_BUCKET, unique_key, b"signed-url-test")
        url = gcs_client.generate_presigned_url(TEST_BUCKET, unique_key, expires_in=300)
        assert isinstance(url, str)
        assert unique_key in url

    def test_large_object(self, gcs_client, unique_key):
        data = b"x" * (1024 * 1024)  # 1 MB
        gcs_client.put_object(TEST_BUCKET, unique_key, data)
        result = gcs_client.get_object(TEST_BUCKET, unique_key)
        assert result == data

    @override_settings(
        STORAGE_BACKEND="gcs",
        GCS_PROJECT_ID="test-project",
        GCS_USE_EMULATOR=True,
        GCS_SIGNING_SERVICE_ACCOUNT="",
        AWS_SBOMS_STORAGE_BUCKET_NAME=TEST_BUCKET,
    )
    def test_storage_client_gcs_integration(self, gcs_client, unique_key):
        """End-to-end test via StorageClient with GCS backend."""
        # Override GCS_ENDPOINT_URL from environment
        with override_settings(GCS_ENDPOINT_URL=os.environ["GCS_ENDPOINT_URL"]):
            client = StorageClient("SBOMS")
            assert isinstance(client._store, GCSObjectStoreClient)

            data = b'{"bomFormat": "CycloneDX", "specVersion": "1.6"}'
            filename = client.upload_sbom(data)
            assert filename.endswith(".json")

            retrieved = client.get_sbom_data(filename)
            assert retrieved == data
