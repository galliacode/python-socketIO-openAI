import { ChatSession } from "../interfaces/ChatSessionInterface";
import { Message } from "../interfaces/MesssageInterface";
import {
  deleteMethod,
  get,
  patchURLEncoded,
  post,
  postURLEncoded,
} from "../utils/APIHelper";
import io from "socket.io-client";

export const getAllChatSessions = async (): Promise<ChatSession[]> => {
  const response = await get(
    `${import.meta.env.VITE_SERVER_ADRESS}/chat_session/sessions`,
  );
  if (response.ok) {
    return response.response.toReversed();
  }
  return [];
};

export const getChatHistry = async (
  chatSessionId: string,
): Promise<Message[]> => {
  const response = await get(
    `${
      import.meta.env.VITE_SERVER_ADRESS
    }/chat_session/${chatSessionId}/history`,
  );
  if (response.ok) {
    return response.response?.history;
  }
  return [];
};

export const createChatSession = async (
  chat_name: string,
): Promise<ChatSession | null> => {
  const response = await post(
    `${
      import.meta.env.VITE_SERVER_ADRESS
    }/chat_session/create?chat_name=${encodeURIComponent(chat_name)}`,
  );
  if (response.ok) {
    return response.response;
  }
  return null;
};

export const sendChatQuery = async ({
  chatSessionId,
  message,
}: {
  chatSessionId: string;
  message: string;
}): Promise<Message | null> => {
  return new Promise(async (resolve, reject) => {
  const body = {
    "user_query": message,
    "token": localStorage.getItem("token"),
    "sessionId": chatSessionId
  };
  const socket = io(`http://localhost:8000`)

  socket.on('connect', () => {
      console.log('Connected to the server');
      socket.emit('message', JSON.stringify(body));
  });
  // const response = await postURLEncoded(
  //   `${import.meta.env.VITE_SERVER_ADRESS}/${chatSessionId}/query`,
  //   body,
  // );
  // if (response.ok) {
  //   resolve(response.response);
  //   return;
  // }
});
};

export const deleteChatSession = async ({
  session_id,
}: {
  session_id: string;
}): Promise<boolean> => {
  const response = await deleteMethod(
    `${import.meta.env.VITE_SERVER_ADRESS}/chat_session/${session_id}/delete`,
  );
  if (response.ok) {
    return true;
  }
  return false;
};

export const renameChatSession = async ({
  session_id,
  title,
}: {
  session_id: string;
  title: string;
}): Promise<boolean> => {
  const response = await patchURLEncoded(
    `${import.meta.env.VITE_SERVER_ADRESS}/chat_session/${session_id}/`,
    {
      title,
    },
  );
  if (response.ok) {
    return true;
  }
  return false;
};

export const submitFeedback = async ({
  session_id,
  history_id,
  user_prompt,
  ai_response,
  user_rating,
  user_comment,
}: {
  session_id: string;
  history_id: number;
  user_prompt: string;
  ai_response: string;
  user_rating: string;
  user_comment: string;
}): Promise<boolean> => {
  const response = await postURLEncoded(
    `${import.meta.env.VITE_SERVER_ADRESS}/${history_id}/feedback`,
    {
      session_id,
      user_prompt,
      ai_response,
      user_rating,
      user_comment,
    },
  );
  if (response.ok) {
    return true;
  }
  throw new Error(
    response.message || "Some error occured while processing user query",
  );
};
