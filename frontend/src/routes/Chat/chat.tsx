import { CircularProgress, IconButton } from "@mui/material";
import TextField from "../../components/TextField/TextField";
import {
  StyledChatHeaderWrapper,
  StyledChatPageWrapper,
  StyledCircularLoader,
  StyledMessageInputWrapper,
  StyledMessagesWrapper,
  StyledSendIcon,
} from "./style";
import Message from "../../components/Message/Message";
import { Message as MesssageInterface } from "../../interfaces/MesssageInterface";
import { useNavigate, useParams } from "react-router-dom";
import { useChatHistory } from "../../hooks/useChatHistory";
import { useForm } from "react-hook-form";
import { useChatSessions } from "../../hooks/useChatSessions";
import { useLayoutEffect, useRef } from "react";

const BOT_AVATAR = "/analitiq-bot.png";

export interface Dataframe {
  columns: string[];
  index: string[];
  data: string[][];
}

const isFollowUp = (messages: MesssageInterface[], index: number) => {
  if (index === 0) return false;
  const previousMessage = messages[index - 1];
  const currentMessage = messages[index];
  if (previousMessage.type !== currentMessage.type) return false;
  return true;
};

const isLastMessageByUser = (messages: MesssageInterface[], index: number) => {
  if (index === messages.length - 1) return true;
  const currentMessage = messages[index];
  const nextMessage = messages[index + 1];
  if (currentMessage.type !== nextMessage.type) return true;
  return false;
};

const getUserQuery = (messages: MesssageInterface[], index: number) => {
  if (index === 0) return "";
  while (index >= 0) {
    if (messages[index].type === "user") return messages[index].content;
    index--;
  }
};

const isFirstMessageByUser = (messages: MesssageInterface[], index: number) => {
  if (index === 0) return true;
  const previousMessage = messages[index - 1];
  const currentMessage = messages[index];
  if (previousMessage.type !== currentMessage.type) return true;
  return false;
};
const ChatPage = () => {
  const { chatId } = useParams();
  const {
    data: messages,
    sendUserQuery,
    isLoading: isChatHistoryLoading,
    sendFeedback,
    sendingFeedback,
  } = useChatHistory(chatId || "");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sendUserQueryRef = useRef(sendUserQuery);
  sendUserQueryRef.current = sendUserQuery;
  const navigate = useNavigate();
  const { createNewChatSessionMutation } = useChatSessions();
  const { register, handleSubmit, watch, setValue } = useForm();
  const messageWatch = watch("message");
  const onSubmit = async (data) => {
    setValue("message", "");
    if (!chatId) {
      try {
        const res = await createNewChatSessionMutation(data.message);
        if (res) {
          navigate(`/chat/${res.id}`);
          setTimeout(() => {
            sendUserQueryRef.current(data.message);
          }, 20);
        }
      } catch (err) {
        console.error(err);
      }
    } else {
      sendUserQuery(data.message);
    }
  };

  useLayoutEffect(() => {
    if (messagesEndRef.current)
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isQueryExecuting = messages
    ? messages[messages.length - 1]?.content.startsWith("<Loader />")
    : false;
  return (
    <>
      {isChatHistoryLoading && (
        <StyledCircularLoader>
          <CircularProgress />
        </StyledCircularLoader>
      )}
      {!isChatHistoryLoading && (
        <StyledChatPageWrapper>
          <StyledMessagesWrapper>
            {messages?.map((message, index) => {
              return (
                <Message
                  key={index}
                  isBotMessage={message.type === "ai"}
                  isFollowUp={isFollowUp(messages, index)}
                  isFirstMessageByUser={isFirstMessageByUser(messages, index)}
                  isLastMessageByUser={isLastMessageByUser(messages, index)}
                  senderName={message.type === "ai" ? "Analitiq" : "You"}
                  senderAvatar={message.type === "ai" ? BOT_AVATAR : ""}
                  content={message.content}
                  type={message.type}
                  contentType={message.content_type}
                  userQuery={getUserQuery(messages, index)}
                  sendFeedback={sendFeedback}
                  sendingFeedback={sendingFeedback}
                  historyId={message.history_id}
                />
              );
            })}
            {(!messages || messages.length === 0) && (
              <StyledChatHeaderWrapper>
                How can I help you today?
              </StyledChatHeaderWrapper>
            )}
            <div ref={messagesEndRef} />
          </StyledMessagesWrapper>
          <StyledMessageInputWrapper>
            <form onSubmit={handleSubmit(onSubmit)}>
              <TextField
                disabled={isQueryExecuting}
                fullWidth
                placeholder="Message Analitiq..."
                multiline
                maxRows={4}
                size="medium"
                InputProps={{
                  sx: {
                    padding: "8px 14px",
                  },
                  endAdornment: (
                    <IconButton
                      sx={{
                        padding: "8px",
                      }}
                      type="submit"
                      disabled={!messageWatch}
                    >
                      <StyledSendIcon />
                    </IconButton>
                  ),
                }}
                onKeyDown={(e) => {
                  if (e.keyCode === 13 && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(onSubmit)();
                  }
                }}
                {...register("message", {
                  required: {
                    message: "Message is required",
                    value: true,
                  },
                })}
              />
            </form>
          </StyledMessageInputWrapper>
        </StyledChatPageWrapper>
      )}
    </>
  );
};

export default ChatPage;
