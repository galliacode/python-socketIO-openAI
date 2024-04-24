import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import {
  getChatHistry,
  sendChatQuery,
  submitFeedback,
} from "../services/chatSessionService";
import { toast } from "react-toastify";

export const useChatHistory = (sessionId: string) => {
  const queryClient = useQueryClient();
  const fetchConversationHistory = useCallback(async () => {
    return await getChatHistry(sessionId);
  }, [sessionId]);
  const { data, isLoading } = useQuery({
    queryKey: ["chatHistory", sessionId],
    queryFn: fetchConversationHistory,
    enabled: !!sessionId,
  });

  const sendQuery = (message: string) => {
    return sendChatQuery({
      chatSessionId: sessionId,
      message,
    });
  };

  const { mutate: sendUserQuery } = useMutation({
    mutationFn: sendQuery,
    onMutate: async (message) => {
      await queryClient.cancelQueries({ queryKey: ["chatHistory", sessionId] });
      const previousHistory = queryClient.getQueryData([
        "chatHistory",
        sessionId,
      ]);
      queryClient.setQueryData(["chatHistory", sessionId], (old: any) => {
        if (old) {
          return [
            ...old,
            {
              content: message,
              content_type: null,
              type: "user",
            },
            {
              content: "<Loader />",
              content_type: null,
              type: "ai",
            },
          ];
        }
        return [
          {
            content: message,
            content_type: null,
            type: "user",
          },
          {
            content: "<Loader />",
            content_type: null,
            type: "ai",
          },
        ];
      });
      return { previousHistory };
    },
    onError: (err: any, newHistory, context) => {
      toast.error(err.message);
      if (context?.previousHistory) {
        queryClient.setQueryData(
          ["chatHistory", sessionId],
          context.previousHistory,
        );
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ["chatHistory", sessionId],
      });
    },
  });

  const { mutateAsync: sendFeedback, isPending: sendingFeedback } = useMutation(
    {
      mutationFn: submitFeedback,
      onError: (err) => {
        toast.error(err.message);
      },
      onSuccess: () => {
        toast.success("Your feedback has been recorded");
      },
    },
  );
  return {
    data: data,
    isLoading,
    sendUserQuery,
    sendFeedback,
    sendingFeedback,
  };
};
