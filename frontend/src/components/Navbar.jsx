function Navbar() {
  const user = JSON.parse(
    localStorage.getItem("user") || "{}"
  );

  return (
    <header className="navbar">
      <div>
        <h3>Knowledge Base Administration</h3>
      </div>

      <div className="navbar-user">
        <span>{user.username || "Administrator"}</span>
      </div>
    </header>
  );
}

export default Navbar;