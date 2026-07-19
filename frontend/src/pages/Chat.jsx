import {useEffect , useState } from "react";
import {
  Send,
  Bot,
  User,
} from "lucide-react";

import apiClient from "../api/apiClient";
import {
  getCurrentUser,
} from "../utils/permissions";


function Chat() {

  const user = getCurrentUser();

  const [messages, setMessages] = useState(() => {
    const savedMessages =
        localStorage.getItem("ekip_chat_messages");

    return savedMessages
        ? JSON.parse(savedMessages)
        : [];
    });


  const [conversationId, setConversationId] =
    useState(() => {

        const savedConversationId =
        localStorage.getItem(
            "ekip_conversation_id"
        );

        if (savedConversationId) {
        return savedConversationId;
        }

        const newConversationId =
        crypto.randomUUID();

        localStorage.setItem(
        "ekip_conversation_id",
        newConversationId
        );

        return newConversationId;
    });

  const [query, setQuery] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const handleSubmit = async (e) => {

    e.preventDefault();

    const question = query.trim();

    if (!question || loading) {
      return;
    }


    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: question,
      },
    ]);

    setQuery("");
    setError("");
    setLoading(true);


    try {

      const response =
        await apiClient.post(
          "/query",
          {
            query: question,

            conversation_id:
              conversationId,
          }
        );


      const data = response.data;


      setMessages((current) => [
        ...current,
        {
          role: "assistant",

          content:
            data.answer,

          sources:
            data.sources || [],

          metadata:
            data.metadata || {},
        },
      ]);


      /*
        If your backend later returns
        conversation_id, store it here.

        Example:

        if (data.conversation_id) {
          setConversationId(
            data.conversation_id
          );
        }
      */

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        "Failed to get response."
      );

    } finally {

      setLoading(false);

    }
  };

  useEffect(() => {
        localStorage.setItem(
            "ekip_chat_messages",
            JSON.stringify(messages)
        );
    }, [messages]);

  const handleNewChat = () => {

    const newConversationId =
        crypto.randomUUID();

    setMessages([]);

    setConversationId(
        newConversationId
    );

    setQuery("");

    setError("");

    localStorage.setItem(
        "ekip_conversation_id",
        newConversationId
    );

    localStorage.removeItem(
        "ekip_chat_messages"
    );
    };
  return (

    <div className="chat-page">

      <div className="page-header">

        <div>

          <h1>
            EKIP Assistant
          </h1>

          <p>
            Ask questions using your
            authorized knowledge base.
          </p>
          
        </div>
        <div className="chat-header-actions">

            <div className="chat-user-info">
            {user.department}
            {" · "}
            {user.role}
            </div>

            <button
            className="secondary-button"
            onClick={handleNewChat}
            >
            New Chat
            </button>

        </div>

      </div>


      <div className="chat-container">

        <div className="messages-container">


          {messages.length === 0 && (

            <div className="empty-chat">

              <Bot size={40} />

              <h2>
                How can I help?
              </h2>

              <p>
                Ask a question about
                your organization's
                knowledge base.
              </p>

            </div>

          )}


          {messages.map(
            (message, index) => (

              <div
                key={index}
                className={
                  message.role === "user"
                    ? "message user-message"
                    : "message assistant-message"
                }
              >

                <div className="message-icon">

                  {message.role ===
                  "user" ? (

                    <User size={20} />

                  ) : (

                    <Bot size={20} />

                  )}

                </div>


                <div className="message-content">

                  <p>
                    {message.content}
                  </p>


                  {message.sources &&
                    message.sources.length >
                      0 && (

                    <div className="sources">

                      <strong>
                        Sources
                      </strong>

                      {message.sources.map(
                        (
                          source,
                          sourceIndex
                        ) => (

                          <div
                            key={
                              sourceIndex
                            }
                            className="source-item"
                          >

                            {typeof source ===
                            "string"
                              ? source
                              : JSON.stringify(
                                  source
                                )}

                          </div>

                        )
                      )}

                    </div>

                  )}

                </div>

              </div>

            )
          )}


          {loading && (

            <div className="message assistant-message">

              <div className="message-icon">

                <Bot size={20} />

              </div>

              <div className="message-content">

                <p>
                  Thinking...
                </p>

              </div>

            </div>

          )}

        </div>


        {error && (

          <div className="alert error-alert">

            {error}

          </div>

        )}


        <form
          className="chat-input-container"
          onSubmit={handleSubmit}
        >

          <textarea
            placeholder="Ask a question..."
            value={query}
            onChange={(e) =>
              setQuery(
                e.target.value
              )
            }
            onKeyDown={(e) => {

              if (
                e.key === "Enter" &&
                !e.shiftKey
              ) {

                e.preventDefault();

                handleSubmit(e);

              }

            }}
          />


          <button
            type="submit"
            disabled={
              loading ||
              !query.trim()
            }
          >

            <Send size={20} />

          </button>

        </form>

      </div>

    </div>

  );
}


export default Chat;