import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Departments from "./pages/Departments";
import Users from "./pages/Users";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import Chat from "./pages/Chat";

function App() {

  return (

    <BrowserRouter>

      <Routes>

        <Route
          path="/"
          element={<Login />}
        />


        <Route
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >

          <Route
            path="/dashboard"
            element={
              <Dashboard />
            }
          />


          <Route
            path="/documents"
            element={
              <ProtectedRoute
                permission="view"
              >
                <Documents />
              </ProtectedRoute>
            }
          />


          <Route
            path="/departments"
            element={
              <ProtectedRoute
                adminOnly
              >
                <Departments />
              </ProtectedRoute>
            }
          />


          <Route
            path="/users"
            element={
              <ProtectedRoute
                adminOnly
              >
                <Users />
              </ProtectedRoute>
            }
          />

          <Route
            path="/chat"
            element={
              <ProtectedRoute
                permission="rag"
              >
                <Chat />
              </ProtectedRoute>
            }
          />

        </Route>

      </Routes>

    </BrowserRouter>

  );
}


export default App;