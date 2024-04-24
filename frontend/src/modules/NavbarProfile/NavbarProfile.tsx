import {
  Avatar,
  Divider,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  MenuList,
  Paper,
} from "@mui/material";
import { StyledNavbarProfileContainer, StyledProfileName } from "./style";
import { Logout, Storage } from "@mui/icons-material";
import React from "react";
import { useAuth } from "../../hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useDatabaseConnection } from "../../hooks/useDatabaseConnection";

const NavbarProfile = () => {
  const { user } = useAuth();
  const { openDatabaseConnectionModal } = useDatabaseConnection();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  const { signout } = useAuth();
  const navigate = useNavigate();
  return (
    <>
      <Menu
        id="basic-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: "top",
          horizontal: "left",
        }}
        transformOrigin={{
          vertical: "bottom",
          horizontal: "left",
        }}
        MenuListProps={{
          "aria-labelledby": "basic-button",
        }}
      >
        <Paper sx={{ width: 236, maxWidth: "100%" }} elevation={0}>
          <MenuList>
            <MenuItem
              onClick={() => {
                openDatabaseConnectionModal();
              }}
            >
              <ListItemIcon>
                <Storage />
              </ListItemIcon>
              <ListItemText>Update DB credentials</ListItemText>
            </MenuItem>
            <Divider />
            <MenuItem
              onClick={() => {
                signout(() => {
                  navigate("/login");
                });
              }}
            >
              <ListItemIcon>
                <Logout fontSize="small" />
              </ListItemIcon>
              <ListItemText>Logout</ListItemText>
            </MenuItem>
          </MenuList>
        </Paper>
      </Menu>
      <StyledNavbarProfileContainer
        fullWidth
        variant="text"
        id="basic-button"
        aria-controls={open ? "basic-menu" : undefined}
        aria-haspopup="true"
        aria-expanded={open ? "true" : undefined}
        onClick={handleClick}
        style={{
          justifyContent: "flex-start",
        }}
      >
        <Avatar sx={{ width: 32, height: 32 }} />
        <StyledProfileName>{user}</StyledProfileName>
      </StyledNavbarProfileContainer>
    </>
  );
};

export default NavbarProfile;
