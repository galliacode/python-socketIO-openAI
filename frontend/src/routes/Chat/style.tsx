import styled from "@emotion/styled";
import { Send } from "@mui/icons-material";

export const StyledChatPageWrapper = styled.div`
  height: 100vh;
  position: relative;
  display: flex;
  flex-direction: column;
  padding: 12px;
`;

export const StyledMessagesWrapper = styled.div`
  padding-top: 40px;
  flex-grow: 1;
  max-height: calc(100vh - 64px);
  overflow: auto;
  margin-bottom: 12px;
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

export const StyledMessageInputWrapper = styled.div`
  & form {
    max-width: 1000px;
    margin: 0 auto;
  }
`;

export const StyledSendIcon = styled(Send)`
  color: ${({ theme }) => (theme as any).palette.primary.main};
`;

export const StyledChatHeader = styled.h1`
  position: absolute;
  padding: 12px 32px;
  background-color: #ffffffeb;
  width: calc(100% - 260px);
  z-index: 1;
`;

export const StyledChatHeaderWrapper = styled.h2`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-weight: 400;
`;

export const StyledCircularLoader = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
`;
