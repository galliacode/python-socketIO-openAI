import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createChatSession,
  deleteChatSession,
  getAllChatSessions,
  renameChatSession,
} from "../services/chatSessionService";

export const useChatSessions = () => {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["chatSessions"],
    queryFn: getAllChatSessions,
  });

  const { mutateAsync: createNewChatSessionMutation } = useMutation({
    mutationFn: createChatSession,
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ["chatSessions"],
      });
    },
  });

  const { mutate: deleteChatSessionMuation } = useMutation({
    mutationFn: deleteChatSession,
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ["chatSessions"],
      });
    },
  });

  const { mutate: renameChatSessionMutation } = useMutation({
    mutationFn: renameChatSession,
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ["chatSessions"],
      });
    },
  });

  return {
    data,
    isLoading,
    createNewChatSessionMutation,
    deleteChatSessionMuation,
    renameChatSessionMutation,
  };
};
