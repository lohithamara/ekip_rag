import {
  NavLink,
  useNavigate,
} from "react-router-dom";

import {
  LayoutDashboard,
  FileText,
  Building2,
  Users,
  MessageSquare,
  BarChart3,
  LogOut,
} from "lucide-react";

import {
  getCurrentUser,
  hasPermission,
  isAdmin,
} from "../utils/permissions";


function Sidebar() {

  const navigate = useNavigate();

  const user = getCurrentUser();

  const logout = () => {

    localStorage.clear();

    navigate("/",{ replace: true });
  };


  return (

    <aside className="sidebar">

      <div className="logo">

        <h2>EKIP</h2>

        <span>
          Knowledge Platform
        </span>

      </div>


      <div className="sidebar-user">

        <strong>
          {user.username}
        </strong>

        <span>
          {user.role}
        </span>

      </div>


      <nav>

        <NavLink to="/dashboard">

          <LayoutDashboard size={20} />

          Dashboard

        </NavLink>


        {hasPermission("view") && (

          <NavLink to="/documents">

            <FileText size={20} />

            Documents

          </NavLink>

        )}


        {isAdmin() && (

          <NavLink to="/departments">

            <Building2 size={20} />

            Departments

          </NavLink>

        )}


        {isAdmin() && (

          <NavLink to="/users">

            <Users size={20} />

            Users

          </NavLink>

        )}


        {hasPermission("rag") && (

          <NavLink to="/chat">

            <MessageSquare size={20} />

            Chat

          </NavLink>

        )}


        {hasPermission("analytics") && (

          <NavLink to="/analytics">

            <BarChart3 size={20} />

            Analytics

          </NavLink>

        )}

      </nav>


      <button
        className="logout-button"
        onClick={logout}
      >

        <LogOut size={20} />

        Logout

      </button>

    </aside>

  );
}

export default Sidebar;