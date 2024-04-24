import { Outlet } from "react-router-dom";
import Navbar from "../../modules/Navbar/Navbar";
import { StyledOutletContainer, StyledRootContainer } from "./style";

const Root = () => {
  return (
    <StyledRootContainer>
      <Navbar />
      <StyledOutletContainer>
        <Outlet />
      </StyledOutletContainer>
    </StyledRootContainer>
  );
};

export default Root;
