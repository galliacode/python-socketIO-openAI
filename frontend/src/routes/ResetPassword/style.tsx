import styled from "@emotion/styled";
import { Paper } from "@mui/material";

export const StyledResetPasswordContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: ${(props) => (props.theme as any).palette.secondary.main};
  min-height: 100vh;
`;

export const StyledResetPasswordCard = styled(Paper)`
  background-color: ${(props) => (props.theme as any).palette.background.paper};
  padding: 30px;
  min-width: 400px;
  border-radius: 20px;
`;
export const StyledResetPasswordCardTitle = styled.h1`
  margin-bottom: 20px;
`;

export const StyledInputContainer = styled.div`
  margin: 12px 0;
`;
