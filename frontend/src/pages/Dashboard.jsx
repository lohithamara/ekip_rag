import { useEffect, useState } from "react";
import {
  FileText,
  Building2,
  Users,
  ShieldCheck,
} from "lucide-react";

import apiClient from "../api/apiClient";
import {
  getCurrentUser,
  isAdmin,
} from "../utils/permissions";

function Dashboard() {
  const user = getCurrentUser();

  const [documents, setDocuments] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const documentResponse =
          await apiClient.get("/documents");

        setDocuments(documentResponse.data);

        const departmentResponse =
          await apiClient.get("/departments");

        setDepartments(
          departmentResponse.data
        );

        if (isAdmin()) {
          const userResponse =
            await apiClient.get("/users");

          setUsers(userResponse.data);
        }
      } catch (error) {
        console.error(
          "Dashboard loading failed:",
          error
        );
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  if (loading) {
    return <p>Loading dashboard...</p>;
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>

          <p>
            Welcome back, {user.username}
          </p>
        </div>
      </div>

      <div className="dashboard-cards">

        <div className="dashboard-card">
          <FileText size={26} />

          <div>
            <span>Documents</span>
            <h2>{documents.length}</h2>
          </div>
        </div>

        <div className="dashboard-card">
          <Building2 size={26} />

          <div>
            <span>
              Departments
            </span>

            <h2>
              {departments.length}
            </h2>
          </div>
        </div>

        {isAdmin() && (
          <div className="dashboard-card">
            <Users size={26} />

            <div>
              <span>Users</span>
              <h2>{users.length}</h2>
            </div>
          </div>
        )}

        <div className="dashboard-card">
          <ShieldCheck size={26} />

          <div>
            <span>Your Role</span>

            <h3>
              {user.role}
            </h3>
          </div>
        </div>

      </div>

      <div className="content-card">
        <h3>Your Access</h3>

        <p>
          <strong>Username:</strong>{" "}
          {user.username}
        </p>

        <p>
          <strong>Role:</strong>{" "}
          {user.role}
        </p>

        <p>
          <strong>Department:</strong>{" "}
          {user.department}
        </p>
      </div>
    </div>
  );
}

export default Dashboard;