import { useLocation, useNavigate, useParams } from "react-router-dom";
import Button from "../../components/Button/Button";
import { useChatSessions } from "../../hooks/useChatSessions";
import NavbarProfile from "../NavbarProfile/NavbarProfile";
import {
  StyledChatMenuItem,
  StyledChatSessionContainer,
  StyledLogoContainer,
  StyledNavbarContainer,
  StyledNavbarProfileWrapper,
  StyledNavbarWrapper,
} from "./style";
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  MenuList,
} from "@mui/material";
import { Add, Create, Delete, MoreVert } from "@mui/icons-material";
import { ChatSession } from "../../interfaces/ChatSessionInterface";
import React from "react";
import { UseMutateFunction } from "@tanstack/react-query";
import TextField from "../../components/TextField/TextField";
import { useForm } from "react-hook-form";

const ChatMenuItem: React.FC<{
  chatSession: ChatSession;
  chatId: string;
  deleteChatSessionMuation: UseMutateFunction<
    boolean,
    Error,
    {
      session_id: string;
    },
    unknown
  >;
  renameChatSessionMutation: UseMutateFunction<
    boolean,
    Error,
    {
      session_id: string;
      title: string;
    },
    unknown
  >;
}> = ({
  chatSession,
  chatId,
  deleteChatSessionMuation,
  renameChatSessionMutation,
}) => {
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const [openDeleteDialog, setOpenDeleteDialog] = React.useState(false);
  const [openEditDialog, setOpenEditDialog] = React.useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues: {
      name: chatSession.chat_name,
    },
  });
  const handleDeleteDialogClose = () => {
    setOpenDeleteDialog(false);
  };

  const handleEditDialogClose = () => {
    setOpenEditDialog(false);
  };
  const handleEditClick = () => {
    setOpenEditDialog(true);
    handleClose();
  };

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleEdit = (data) => {
    renameChatSessionMutation({
      session_id: chatSession.id,
      title: data.name,
    });
    handleClose();
  };

  const handleDeleteClick = () => {
    handleClose();
    setOpenDeleteDialog(true);
  };

  const handleDelete = () => {
    if (chatId === chatSession.id) {
      navigate(`/chat`);
    }
    deleteChatSessionMuation({
      session_id: chatSession.id,
    });
    handleDeleteDialogClose();
  };
  return (
    <StyledChatMenuItem>
      <MenuItem
        key={chatSession.id}
        onClick={() => {
          navigate(`/chat/${chatSession.id}`);
        }}
        sx={{
          maxWidth: "240px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          flexGrow: 1,
        }}
      >
        <ListItemText
          primaryTypographyProps={{
            sx: {
              overflow: "hidden",
              textOverflow: "ellipsis",
            },
          }}
        >
          <span
            style={{
              fontWeight: chatSession.id === chatId ? "bold" : "normal",
            }}
          >
            {chatSession.chat_name}
          </span>
        </ListItemText>
      </MenuItem>
      <IconButton onClick={handleClick}>
        <MoreVert />
      </IconButton>
      <Menu
        id="long-menu"
        MenuListProps={{
          "aria-labelledby": "long-button",
        }}
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
      >
        <MenuItem onClick={handleDeleteClick}>
          <ListItemIcon>
            <Delete fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Delete" />
        </MenuItem>
      </Menu>
      <Dialog
        open={openDeleteDialog}
        onClose={handleDeleteDialogClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">Delete Chat Session</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Are you sure you want to delete this chat session?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={handleDeleteDialogClose}>
            Cancel
          </Button>
          <Button onClick={handleDelete} autoFocus>
            Continue
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={openEditDialog} onClose={handleEditDialogClose}>
        <form onSubmit={handleSubmit(handleEdit)}>
          <DialogTitle>Rename</DialogTitle>
          <DialogContent>
            <DialogContentText>
              To rename to this chat session, please enter new chat session
              name.
            </DialogContentText>
            <TextField
              autoFocus
              margin="dense"
              id="name"
              label="Name"
              type="text"
              fullWidth
              variant="standard"
              helperText={errors.name ? (errors.name.message as string) : null}
              error={errors.name ? true : false}
              {...register("name", {
                required: {
                  message: "name is required",
                  value: true,
                },
              })}
            />
          </DialogContent>
          <DialogActions>
            <Button
              type="button"
              variant="outlined"
              onClick={handleEditDialogClose}
            >
              Cancel
            </Button>
            <Button type="submit">Rename</Button>
          </DialogActions>
        </form>
      </Dialog>
    </StyledChatMenuItem>
  );
};

const ChatSessionHolder = () => {
  const {
    data: chatSessions,
    isLoading: chatSessionsLoading,
    deleteChatSessionMuation,
    renameChatSessionMutation,
  } = useChatSessions();
  const { chatId } = useParams();
  const navigate = useNavigate();

  return (
    <StyledChatSessionContainer>
      {chatSessionsLoading ? (
        <div>Loading...</div>
      ) : (
        <>
          <MenuList>
            <MenuItem
              onClick={() => {
                navigate(`/chat`);
              }}
            >
              <ListItemText>New Chat</ListItemText>
              <ListItemIcon
                sx={{
                  marginRight: "-20px",
                }}
              >
                <Add />
              </ListItemIcon>
            </MenuItem>
            {chatSessions?.map((chatSession) => (
              <ChatMenuItem
                chatId={chatId || ""}
                chatSession={chatSession}
                key={chatSession.id}
                deleteChatSessionMuation={deleteChatSessionMuation}
                renameChatSessionMutation={renameChatSessionMutation}
              />
            ))}
          </MenuList>
        </>
      )}
    </StyledChatSessionContainer>
  );
};

const Navbar = () => {
  return (
    <StyledNavbarWrapper>
      <StyledNavbarContainer>
        <StyledLogoContainer>
          <img
            src="http://try.analitiq.ai/media/f033ee18ed21c9c90af74333517413ba6aee5dd575e4f7eb183637be.png"
            alt="logo"
          />
        </StyledLogoContainer>
        <ChatSessionHolder />
        <StyledNavbarProfileWrapper>
          <NavbarProfile />
        </StyledNavbarProfileWrapper>
      </StyledNavbarContainer>
    </StyledNavbarWrapper>
  );
};

export default Navbar;
