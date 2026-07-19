import { useEffect, useState } from "react";
import {
  Building2,
  Plus,
  Trash2,
  RefreshCw,
} from "lucide-react";

import apiClient from "../api/apiClient";

function Departments() {
  const [departments, setDepartments] = useState([]);
  const [name, setName] = useState("");

  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const fetchDepartments = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await apiClient.get(
        "/departments"
      );

      setDepartments(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to load departments."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDepartments();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();

    if (!name.trim()) {
      setError("Enter a department name.");
      return;
    }

    setCreating(true);
    setError("");
    setMessage("");

    try {
      await apiClient.post(
        "/departments",
        {
          name: name.trim(),
        }
      );

      setName("");

      setMessage(
        "Department created successfully."
      );

      await fetchDepartments();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to create department."
      );
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (
    departmentId
  ) => {
    const confirmed = window.confirm(
      "Are you sure you want to delete this department?"
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setMessage("");

    try {
      await apiClient.delete(
        `/departments/${departmentId}`
      );

      setMessage(
        "Department deleted successfully."
      );

      await fetchDepartments();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to delete department."
      );
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Departments</h1>

          <p>
            Manage departments within your
            organization.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={fetchDepartments}
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

      <div className="content-card">
        <h3>Create Department</h3>

        <form
          className="upload-form"
          onSubmit={handleCreate}
        >
          <input
            type="text"
            placeholder="Department name"
            value={name}
            onChange={(e) =>
              setName(e.target.value)
            }
          />

          <button
            type="submit"
            className="primary-button"
            disabled={creating}
          >
            <Plus size={18} />

            {creating
              ? "Creating..."
              : "Create"}
          </button>
        </form>
      </div>

      <div className="content-card">
        <div className="table-header">
          <h3>
            <Building2
              size={20}
              style={{
                verticalAlign: "middle",
                marginRight: "8px",
              }}
            />

            Departments
          </h3>
        </div>

        {loading ? (
          <p>Loading departments...</p>
        ) : departments.length === 0 ? (
          <p>No departments found.</p>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Department</th>
                  <th>Actions</th>
                </tr>
              </thead>

              <tbody>
                {departments.map(
                  (department) => (
                    <tr key={department.id}>
                      <td>
                        {department.id}
                      </td>

                      <td>
                        {department.name}
                      </td>

                      <td className="actions">
                        <button
                          title="Delete"
                          onClick={() =>
                            handleDelete(
                              department.id
                            )
                          }
                        >
                          <Trash2
                            size={17}
                          />
                        </button>
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Departments;