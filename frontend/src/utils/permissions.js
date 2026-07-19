export const ROLE_PERMISSIONS = {
  admin: ["*"],

  knowledge_admin: [
    "view",
    "upload",
    "delete",
    "reindex",
    "rag",
    "download",
  ],

  department_manager: [
    "view",
    "rag",
    "download",
    "analytics",
  ],

  employee: [
    "view",
    "rag",
    "download",
  ],

  auditor: [
    "view",
    "rag",
    "download",
  ],
};

export function getCurrentUser() {
  try {
    return JSON.parse(
      localStorage.getItem("user") || "{}"
    );
  } catch {
    return {};
  }
}

export function hasPermission(permission) {
  const user = getCurrentUser();

  const permissions =
    ROLE_PERMISSIONS[user.role] || [];

  return (
    permissions.includes("*") ||
    permissions.includes(permission)
  );
}

export function isAdmin() {
  return getCurrentUser().role === "admin";
}