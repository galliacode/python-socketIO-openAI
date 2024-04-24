import styled from "@emotion/styled";
import DataEditor from "@glideapps/glide-data-grid";
import { Paper } from "@mui/material";

interface StyledMessageWrapperProps {
  isLastMessageByUser: boolean;
  isBotMessage: boolean;
  isFirstMessageByUser: boolean;
}

interface StyledContentContainerProps {
  isBotMessage: boolean;
}
export const StyledMessageWrapper = styled(Paper)<StyledMessageWrapperProps>`
  /* background-color: ${(props) =>
    !props.isBotMessage
      ? (props.theme as any).palette.secondary.main
      : null}; */
  margin: 0 auto;
  margin-top: ${(props) => (props.isFirstMessageByUser ? "16px" : "0px")};
  display: flex;
  padding: 16px;
  padding-bottom: ${(props) => (props.isLastMessageByUser ? "16px" : "6px")};
  padding-top: ${(props) => (props.isFirstMessageByUser ? "16px" : "0px")};
  max-width: 1000px;
  position: relative;
  &:hover {
    & > .message-feedback {
      display: flex !important;
    }
  }
`;

export const StyledSenderName = styled.div`
  font-weight: 600;
  font-size: 16px;
  margin: 0px 0 8px 0;
`;

export const StyledContentContainer = styled.div<StyledContentContainerProps>`
  width: calc(100% - 40px);
  min-height: ${(props) => (props.isBotMessage ? "105px" : "0px")};
`;

export const StyledInstruction = styled.p`
  color: ${(props) => (props.theme as any).palette.primary.main};
  font-weight: 600;
  margin: 8px 0;
`;
export const StyledAvatarContainer = styled.div`
  min-width: 40px;
`;

export const StyledIframe = styled.iframe`
  border: none;
`;

export const StyledDataEditor = styled(DataEditor)`
  ::-webkit-scrollbar {
    width: 10px; /* width of the scrollbar */
  }

  /* Track */
  ::-webkit-scrollbar-track {
    background: #f1f1f1; /* color of the track */
  }

  /* Handle */
  ::-webkit-scrollbar-thumb {
    background: #88888831; /* color of the scrollbar */
    border-radius: 5px; /* roundness of the scrollbar */
  }

  /* Firefox scrollbar */
  /* Works in Firefox */
  /* Firefox supports scrollbar-color and scrollbar-width properties */
  /* These properties are experimental, so browser support may vary */

  /* Color of the scrollbar */
  /* The first value is the thumb, the second is the track */
  /* Only works in Firefox Nightly with layout.css.scrollbar-color.enabled = true in about:config */
  scrollbar-color: #888 #f1f1f1;

  /* Width of the scrollbar */
  /* Only works in Firefox Nightly with layout.css.scrollbar-width.enabled = true in about:config */
  scrollbar-width: thin; /* auto | thin | none | normal */
`;

export const StyledDataframeContainer = styled.div`
  min-width: 52px;
  min-height: 73px;
  box-sizing: border-box;
  border-radius: 0.5rem;
  border: 1px solid rgba(0, 0, 0, 0.1);
  margin: 6px 0;
`;

export const StyledMessageLoader = styled.div`
  display: flex;
  align-items: center;
  margin: 6px 0;
  & > span {
    margin-left: 8px;
  }
`;

export const StyledText = styled.div`
  line-height: 1.714;
  letter-spacing: 0.005em;
`;

export const StyledFeedbackForm = styled.form`
  width: 600px;
`;
export const StyledFeedbackFormContainer = styled.div`
  display: flex;
  align-items: center;
`;

export const StyledFeedbackIconBackground = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 50px;
  height: 50px;
  margin: 5px;
  margin-right: 20px;
  overflow: hidden;
  border-radius: 50%;
  vertical-align: middle;
`;
export const StyledFeedbackTitle = styled.span`
  vertical-align: middle;
`;

export const StyledCheckboxContainer = styled.div`
  margin: -15px 0;
`;
