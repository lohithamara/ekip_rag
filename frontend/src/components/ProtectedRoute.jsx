import {
  Navigate,
} from "react-router-dom";

import {
  getCurrentUser,
  hasPermission,
} from "../utils/permissions";


function ProtectedRoute({
  children,
  permission,
  adminOnly = false,
}) {

  const token =
    localStorage.getItem(
      "access_token"
    );

  const user =
    getCurrentUser();


  if (!token) {

    return (
      <Navigate
        to="/"
        replace
      />
    );

  }


  if (
    adminOnly &&
    user.role !== "admin"
  ) {

    return (
      <Navigate
        to="/dashboard"
        replace
      />
    );

  }


  if (
    permission &&
    !hasPermission(permission)
  ) {

    return (
      <Navigate
        to="/dashboard"
        replace
      />
    );

  }


  return children;
}


export default ProtectedRoute;