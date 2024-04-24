import styled from "@emotion/styled";

export const StyledNavbarContainer = styled.div`
  width: inherit;
  position: fixed;
  top: 0;
  height: 100vh;
  padding: 12px 12px 0 12px;
  background-color: ${(props) => (props.theme as any).palette.secondary.main};
  display: flex;
  flex-direction: column;
  justify-content: space-between;
`;

export const StyledNavbarWrapper = styled.div`
  min-height: 100vh;
  width: 260px;
  min-width: 260px;
  position: relative;
`;

export const StyledLogoContainer = styled.div`
  width: 100px;
  margin: auto;
  img {
    width: 100%;
  }
`;

export const StyledNavbarProfileWrapper = styled.div`
  width: calc(100% - 24px);
  margin: 12px;
`;

export const StyledChatSessionContainer = styled.div`
  flex-grow: 1;
`;

export const StyledChatMenuItem = styled.div`
  display: flex;
`;
