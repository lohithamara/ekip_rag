import { useEffect, useState } from "react";
import {
  UserPlus,
  RefreshCw,
} from "lucide-react";

import apiClient from "../api/apiClient";

const ROLES = [
  "admin",
  "knowledge_admin",
  "department_manager",
  "employee",
  "auditor",
];

function Users() {
  const [users, setUsers] = useState([]);
  const [departments, setDepartments] = useState([]);

  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    role: "employee",
    department: "",
  });

  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const fetchUsers = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await apiClient.get("/users");
      setUsers(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to load users."
      );
    } finally {
      setLoading(false);
    }
  };

  const fetchDepartments = async () => {
    try {
      const response = await apiClient.get(
        "/departments"
      );

      setDepartments(response.data);

      if (
        response.data.length > 0 &&
        !form.department
      ) {
        setForm((current) => ({
          ...current,
          department: response.data[0].name,
        }));
      }
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to load departments."
      );
    }
  };

  useEffect(() => {
    fetchUsers();
    fetchDepartments();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  };

  const handleCreate = async (e) => {
    e.preventDefault();

    setCreating(true);
    setError("");
    setMessage("");

    try {
      await apiClient.post("/users", form);

      setMessage(
        "User created successfully."
      );

      setForm({
        username: "",
        email: "",
        password: "",
        role: "employee",
        department:
          departments[0]?.name || "",
      });

      await fetchUsers();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to create user."
      );
    } finally {
      setCreating(false);
    }
  };

  const updateUser = async (
    userId,
    changes
  ) => {
    setError("");
    setMessage("");

    try {
      await apiClient.patch(
        `/users/${userId}`,
        changes
      );

      setMessage(
        "User updated successfully."
      );

      await fetchUsers();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to update user."
      );
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Users</h1>
          <p>
            Manage users, roles and department
            assignments.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={fetchUsers}
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
        <h3>Create User</h3>

        <form
          className="user-form"
          onSubmit={handleCreate}
        >
          <input
            name="username"
            placeholder="Username"
            value={form.username}
            onChange={handleChange}
            required
          />

          <input
            name="email"
            type="email"
            placeholder="Email"
            value={form.email}
            onChange={handleChange}
            required
          />

          <input
            name="password"
            type="password"
            placeholder="Password"
            value={form.password}
            onChange={handleChange}
            required
          />

          <select
            name="role"
            value={form.role}
            onChange={handleChange}
          >
            {ROLES.map((role) => (
              <option
                key={role}
                value={role}
              >
                {role}
              </option>
            ))}
          </select>

          <select
            name="department"
            value={form.department}
            onChange={handleChange}
            required
          >
            {departments.map(
              (department) => (
                <option
                  key={department.id}
                  value={department.name}
                >
                  {department.name}
                </option>
              )
            )}
          </select>

          <button
            type="submit"
            className="primary-button"
            disabled={creating}
          >
            <UserPlus size={18} />

            {creating
              ? "Creating..."
              : "Create User"}
          </button>
        </form>
      </div>

      <div className="content-card">
        <h3>System Users</h3>

        {loading ? (
          <p>Loading users...</p>
        ) : users.length === 0 ? (
          <p>No users found.</p>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Department</th>
                  <th>Status</th>
                </tr>
              </thead>

              <tbody>
                {users.map((user) => (
                  <tr key={user.id}>
                    <td>
                      {user.username}
                    </td>

                    <td>
                      {user.email}
                    </td>

                    <td>
                      <select
                        value={user.role}
                        onChange={(e) =>
                          updateUser(
                            user.id,
                            {
                              role:
                                e.target
                                  .value,
                            }
                          )
                        }
                      >
                        {ROLES.map(
                          (role) => (
                            <option
                              key={role}
                              value={role}
                            >
                              {role}
                            </option>
                          )
                        )}
                      </select>
                    </td>

                    <td>
                      <select
                        value={
                          user.department
                        }
                        onChange={(e) =>
                          updateUser(
                            user.id,
                            {
                              department:
                                e.target
                                  .value,
                            }
                          )
                        }
                      >
                        {departments.map(
                          (department) => (
                            <option
                              key={
                                department.id
                              }
                              value={
                                department.name
                              }
                            >
                              {
                                department.name
                              }
                            </option>
                          )
                        )}
                      </select>
                    </td>

                    <td>
                      <button
                        className={
                          user.is_active
                            ? "status-active"
                            : "status-inactive"
                        }
                        onClick={() =>
                          updateUser(
                            user.id,
                            {
                              is_active:
                                !user.is_active,
                            }
                          )
                        }
                      >
                        {user.is_active
                          ? "Active"
                          : "Inactive"}
                      </button>
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

export default Users;