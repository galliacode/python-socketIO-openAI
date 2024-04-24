import React, { useEffect, useRef } from "react";
import {
  StyledAvatarContainer,
  StyledCheckboxContainer,
  StyledContentContainer,
  StyledDataEditor,
  StyledDataframeContainer,
  StyledFeedbackForm,
  StyledFeedbackFormContainer,
  StyledFeedbackIconBackground,
  StyledFeedbackTitle,
  StyledIframe,
  StyledMessageLoader,
  StyledMessageWrapper,
  StyledSenderName,
  StyledText,
} from "./style";
import {
  Avatar,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  useTheme,
} from "@mui/material";
import ThumbUpAltOutlinedIcon from "@mui/icons-material/ThumbUpAltOutlined";
import ThumbDownAltOutlinedIcon from "@mui/icons-material/ThumbDownAltOutlined";
import { Dataframe } from "../../routes/Chat/chat";
import {
  GridCell,
  GridCellKind,
  GridColumn,
  Item,
} from "@glideapps/glide-data-grid";
import CodeCopy from "../CodeCopy/CodeCopy";
import styled from "@emotion/styled";
import { Close } from "@mui/icons-material";
import TextField from "../TextField/TextField";
import Button from "../Button/Button";
import { useForm } from "react-hook-form";
import { UseMutateAsyncFunction } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

interface MessageProps {
  isBotMessage: boolean;
  isFollowUp: boolean;
  content: string;
  senderName: string;
  senderAvatar: string;
  type: string;
  contentType: string;
  isLastMessageByUser: boolean;
  isFirstMessageByUser: boolean;
  userQuery: string;
  sendFeedback: UseMutateAsyncFunction<
    boolean,
    Error,
    {
      session_id: string;
      history_id: number;
      user_prompt: string;
      ai_response: string;
      user_rating: string;
      user_comment: string;
    },
    unknown
  >;
  sendingFeedback: boolean;
  historyId: number;
}

const Grid: React.FC<{ dataframe: Dataframe }> = ({ dataframe }) => {
  const numRows = dataframe.data.length;
  const columns: GridColumn[] = dataframe.columns.map((column) => {
    return {
      title: column,
      width: 200,
    };
  });

  const getData = ([col, row]: Item): GridCell => {
    const rowData = dataframe.data[row];
    const cellData = rowData[col];
    return {
      kind: GridCellKind.Text,
      data: cellData?.toString() || "null",
      allowOverlay: false,
      displayData: cellData?.toString() || "null",
    };
  };
  const ROW_HEIGHT = 37;
  const height =
    dataframe.data.length > 10
      ? ROW_HEIGHT * 11
      : ROW_HEIGHT * (dataframe.data.length + 1);
  return (
    <StyledDataframeContainer>
      <StyledDataEditor
        columnSelect="none"
        getCellContent={getData}
        columns={columns}
        rows={numRows}
        height={height}
        rowMarkers="number"
        width={"100%"}
      />
    </StyledDataframeContainer>
  );
};

export const ContentParser: React.FC<{
  content: string;
  type: string;
  isBotMessage: boolean;
}> = ({ content, type, isBotMessage }) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  useEffect(() => {
    if (iframeRef.current && content.startsWith("<div")) {
      iframeRef.current.contentDocument?.write(content);
    }
  }, [iframeRef.current]);
  if (type === "html") {
    const data = JSON.parse(content);
    if (data.data.length === 0) return null;
    return <Grid dataframe={data} />;
  } else if (type === "javascript") {
    if (!content) return null;
    return (
      <div>
        <StyledIframe
          ref={iframeRef}
          width={"100%"}
          height={500}
        ></StyledIframe>
      </div>
    );
  } else if (type === "text") {
    return <StyledText>{content}</StyledText>;
  } else if (content.startsWith("<Loader />")) {
    return (
      <StyledMessageLoader>
        <CircularProgress size={24} />
        <span>executing query...</span>
      </StyledMessageLoader>
    );
  } else if (isBotMessage && !!content) {
    return <CodeCopy code={content} language="sql" />;
  } else if (!!content) {
    return <StyledText>{content}</StyledText>;
  } else {
    return null;
  }
};
const StyledClosedButton = styled(IconButton)`
  position: absolute;
  right: 5px;
  top: 5px;
`;
const Message: React.FC<MessageProps> = (props) => {
  const [openFeedback, setOpenFeedbck] = React.useState(false);
  const [feedbackType, setFeedbackType] = React.useState<
    "like" | "dislike" | "none"
  >("none");
  const theme = useTheme();
  const { chatId } = useParams();
  const handleOpen = (feedbackType: "like" | "dislike") => {
    setFeedbackType(feedbackType);
    setOpenFeedbck(true);
  };

  const { register, handleSubmit, reset } = useForm({
    defaultValues: {
      harmful: false,
      lies: false,
      unhelpful: false,
      message: "",
    },
  });
  const handleClose = () => {
    reset();
    setOpenFeedbck(false);
  };
  const handleFormSubmit = async (data) => {
    let message = `
    ${data.message}
    ${data.harmful ? "This is harmful / unsafe" : ""}
    ${data.lies ? "This isn't true" : ""}
    ${data.unhelpful ? "This isn't helpful" : ""}
    `;
    message = message.trim();
    await props.sendFeedback({
      user_comment: message,
      user_rating: feedbackType === "like" ? "good" : "bad",
      user_prompt: props.userQuery,
      ai_response: props.content,
      session_id: chatId,
      history_id: props.historyId,
    });
    handleClose();
  };
  return (
    <StyledMessageWrapper
      isBotMessage={props.isBotMessage}
      isLastMessageByUser={props.isLastMessageByUser}
      isFirstMessageByUser={props.isFirstMessageByUser}
      elevation={0}
    >
      {openFeedback && (
        <Dialog
          open={openFeedback}
          onClose={handleClose}
          disableEscapeKeyDown={true}
          slotProps={{
            backdrop: {
              sx: {
                pointerEvents: "none",
              },
            },
          }}
        >
          <StyledFeedbackForm onSubmit={handleSubmit(handleFormSubmit)}>
            <>
              <StyledFeedbackFormContainer>
                {feedbackType === "like" && (
                  <>
                    <DialogTitle>
                      <StyledFeedbackIconBackground
                        style={{
                          backgroundColor: `${theme.palette.success.light}55`,
                        }}
                      >
                        <ThumbUpAltOutlinedIcon color={"success"} />
                      </StyledFeedbackIconBackground>
                      <StyledFeedbackTitle>
                        Provide additional feedback
                      </StyledFeedbackTitle>
                    </DialogTitle>
                  </>
                )}
                {feedbackType === "dislike" && (
                  <>
                    <DialogTitle>
                      <StyledFeedbackIconBackground
                        style={{
                          backgroundColor: `${theme.palette.error.light}55`,
                        }}
                      >
                        <ThumbDownAltOutlinedIcon color={"error"} />
                      </StyledFeedbackIconBackground>
                      <StyledFeedbackTitle>
                        Provide additional feedback
                      </StyledFeedbackTitle>
                    </DialogTitle>
                  </>
                )}
                <StyledClosedButton onClick={handleClose}>
                  <Close />
                </StyledClosedButton>
              </StyledFeedbackFormContainer>
              <Divider />
              <DialogContent>
                <TextField
                  placeholder={
                    feedbackType === "like"
                      ? "What do you like about the response?"
                      : "What was the issue with the response? How could it be improved?"
                  }
                  fullWidth
                  multiline
                  maxRows={4}
                  minRows={4}
                  variant="outlined"
                  {...register("message")}
                />

                {feedbackType === "dislike" && (
                  <>
                    <StyledCheckboxContainer
                      style={{
                        marginTop: "5px",
                      }}
                    >
                      <FormControlLabel
                        control={<Checkbox {...register("harmful")} />}
                        label="This is harmful / unsafe"
                      />
                    </StyledCheckboxContainer>
                    <StyledCheckboxContainer>
                      <FormControlLabel
                        control={<Checkbox {...register("lies")} />}
                        label="This isn't true"
                      />
                    </StyledCheckboxContainer>
                    <StyledCheckboxContainer>
                      <FormControlLabel
                        control={<Checkbox {...register("unhelpful")} />}
                        label="This isn't helpful"
                      />
                    </StyledCheckboxContainer>
                  </>
                )}
              </DialogContent>
              <DialogActions
                sx={{
                  paddingBottom: "20px",
                }}
              >
                <Button
                  type="submit"
                  variant="outlined"
                  disabled={props.sendingFeedback}
                  color={feedbackType === "dislike" ? "error" : "primary"}
                >
                  {props.sendingFeedback
                    ? "submitting feedback..."
                    : "Submit feedback"}
                </Button>
              </DialogActions>
            </>
          </StyledFeedbackForm>
        </Dialog>
      )}
      <StyledAvatarContainer>
        {!props.isFollowUp && (
          <Avatar sx={{ width: 32, height: 32 }} src={props.senderAvatar} />
        )}
      </StyledAvatarContainer>
      <StyledContentContainer isBotMessage={props.isBotMessage}>
        {!props.isFollowUp && (
          <StyledSenderName>{props.senderName}</StyledSenderName>
        )}
        <ContentParser
          key={props.content}
          content={props.content}
          type={props.contentType}
          isBotMessage={props.isBotMessage}
        />
      </StyledContentContainer>
      {props.isBotMessage && (
        <div
          className="message-feedback"
          style={{
            display: "none",
            flexDirection: "column",
            justifyContent: "flex-start",
            alignItems: "center",
            height: "100%",
            position: "absolute",
            top: !props.isFollowUp ? "45px" : "0px",
          }}
        >
          <IconButton onClick={() => handleOpen("like")}>
            <ThumbUpAltOutlinedIcon />
          </IconButton>
          <IconButton onClick={() => handleOpen("dislike")}>
            <ThumbDownAltOutlinedIcon />
          </IconButton>
        </div>
      )}
    </StyledMessageWrapper>
  );
};

export default Message;
