import styled from "@emotion/styled";

export const StyledCodeCopyContainer = styled.div`
  position: relative;
  & pre {
    background-color: #f4f4f4 !important;
  }
  & code {
    min-height: 64px;
  }
`;

export const StyledCopyButton = styled.div`
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 1;
`;
