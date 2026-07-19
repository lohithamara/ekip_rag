import {
  hasPermission,
} from "../utils/permissions";

import { useEffect, useState } from "react";
import {
  Upload,
  Trash2,
  Download,
  RefreshCw,
} from "lucide-react";

import apiClient from "../api/apiClient";

function Documents() {
  const user = JSON.parse(
    localStorage.getItem("user") || "{}"
  );

  const [documents, setDocuments] = useState([]);
  const [file, setFile] = useState(null);
  const [department, setDepartment] = useState("");
  const [filterDepartment, setFilterDepartment] = useState("");

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const tenantId = user.tenant_id;

  const canUpload = hasPermission("upload");

  const canDelete = hasPermission("delete");

  const canDownload = hasPermission("download");

  const fetchDocuments = async () => {
    if (!tenantId) {
      setError("Tenant information not found.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const params = {};

      if (filterDepartment) {
        params.department = filterDepartment;
      }

      const response = await apiClient.get(
        "/documents",
        { params }
      );

      setDocuments(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to load documents."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [filterDepartment]);

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file || !department) {
      setError(
        "Select a file and enter a department."
      );
      return;
    }

    setUploading(true);
    setError("");
    setMessage("");

    const formData = new FormData();

    formData.append("file", file);
    formData.append("department", department);

    try {
      await apiClient.post(
        "/documents/upload",
        formData
      );

      setMessage(
        "Document uploaded. Processing has started."
      );

      setFile(null);

      await fetchDocuments();

    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Document upload failed."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId) => {
    const confirmed = window.confirm(
      "Are you sure you want to delete this document?"
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setMessage("");

    try {
      await apiClient.delete(
        `/documents/${documentId}`
      );

      setDocuments((current) =>
        current.filter(
          (document) =>
            document.document_id !== documentId
        )
      );

      setMessage("Document deleted.");

    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to delete document."
      );
    }
  };

  const handleDownload = async (
    documentId,
    filename
  ) => {
    try {
      const response = await apiClient.get(
        `/documents/download/${documentId}`,
        {
          responseType: "blob",
        }
      );

      const url = window.URL.createObjectURL(
        new Blob([response.data])
      );

      const link =
        document.createElement("a");

      link.href = url;
      link.download = filename || "document";

      document.body.appendChild(link);

      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);

    } catch (err) {
      setError("Failed to download document.");
    }
  };

  return (
    <div className="documents-page">

      <div className="page-header">
        <div>
          <h1>Documents</h1>
          <p>
            Manage knowledge base documents.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={fetchDocuments}
        >
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="alert error-alert">
          {error}
        </div>
      )}

      {message && (
        <div className="alert success-alert">
          {message}
        </div>
      )}
    {canUpload && ( 
        <div className="content-card">

        <h3>Upload Document</h3>

        <form
          className="upload-form"
          onSubmit={handleUpload}
        >

          <input
            type="file"
            onChange={(e) =>
              setFile(e.target.files[0])
            }
          />

          <input
            type="text"
            placeholder="Department"
            value={department}
            onChange={(e) =>
              setDepartment(e.target.value)
            }
          />

          <button
            className="primary-button"
            type="submit"
            disabled={uploading}
          >
            <Upload size={18} />

            {uploading
              ? "Uploading..."
              : "Upload"}
          </button>

        </form>
      </div>)
      }

      <div className="content-card">

        <div className="table-header">

          <h3>Knowledge Base</h3>

          <input
            type="text"
            placeholder="Filter by department"
            value={filterDepartment}
            onChange={(e) =>
              setFilterDepartment(
                e.target.value
              )
            }
          />

        </div>

        {loading ? (
          <p>Loading documents...</p>
        ) : documents.length === 0 ? (
          <p>No documents found.</p>
        ) : (
          <div className="table-container">

            <table>

              <thead>
                <tr>
                  <th>ID</th>
                  <th>Filename</th>
                  <th>Department</th>
                  <th>Status</th>
                  <th>Version</th>
                  <th>Actions</th>
                </tr>
              </thead>

              <tbody>

                {documents.map((document) => (

                  <tr key={document.document_id}>

                    <td>
                      {document.document_id}
                    </td>

                    <td>
                      {document.filename}
                    </td>

                    <td>
                      {document.department}
                    </td>

                    <td>
                      <span
                        className={`status-badge ${
                          document.status || ""
                        }`}
                      >
                        {document.status}
                      </span>
                    </td>

                    <td>
                      {document.version}
                    </td>

                    <td className="actions">

                      {canDownload && (
                        <button
                            title="Download"
                            onClick={() =>
                            handleDownload(
                                document.document_id,
                                document.filename
                            )
                            }
                        >
                            <Download size={17} />
                        </button>
                    )}

                      {canDelete && (
                        <button
                            title="Delete"
                            onClick={() =>
                            handleDelete(
                                document.document_id
                            )
                            }
                        >
                            <Trash2 size={17} />
                        </button>
                     )}

                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>
        )}

      </div>

    </div>
  );
}

export default Documents;