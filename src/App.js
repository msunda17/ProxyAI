import { useState } from "react";
import { Card, CardContent, CardActions, Typography, Box, Paper } from "@mui/material";
import { Button } from "@mui/material";
import { TextField } from "@mui/material";
import { CloudUpload } from "@mui/icons-material";
import JSONPretty from 'react-json-pretty';
import 'react-json-pretty/themes/monikai.css';
import { alignProperty } from "@mui/material/styles/cssUtils";

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [aiMessage, setAiMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const url = "https://0.0.0.0:8000/"

  const handleSubmit = async () => {
    setLoading(true);
    const formData = new FormData();
    if (file) formData.append("file", file);
    if (input) formData.append("url", input);

    try {
      const res = await fetch(url + "generate_activity", {
        method: "POST",
        body: JSON.stringify({ url: input }),
        headers: {
          "Content-Type": "application/json"
        }
      });
      const data = await res.json();
      
      if (typeof data.ai_message === "string") {
        setAiMessage(data.ai_message);
      }
      setResponse(data.structured_response);
    } catch (error) {
      console.error("Error fetching response", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4} sx={{ backgroundColor: "#f4f6f8", minHeight: "100vh" }}>
      <Card sx={{ width: 1000, padding: 3, boxShadow: 3, backgroundColor: "white", justifyContent: "center", alignItems: "center"}}>
        <CardContent>
          <Typography variant="h5" fontWeight="bold" gutterBottom>
            Proxy AI
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            Provide a URL or upload a file to generate structured activity data.
          </Typography>
          <Box mt={2}>
            <TextField
              label="Enter a URL"
              variant="outlined"
              fullWidth
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          </Box>
          {/* <Box mt={2}>
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUpload />}
            >
              Upload CSV/Excel File
              <input
                type="file"
                hidden
                onChange={(e) => setFile(e.target.files[0])}
              />
            </Button>
          </Box> */}
          <CardActions>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? "Processing..." : "Generate Activity"}
            </Button> 
          </CardActions>
        </CardContent>
      </Card>

      {response && (
        <Paper elevation={3} sx={{ maxWidth: 700, mt: 3, padding: 2, backgroundColor: "#fafafa" }}>
          <Typography variant="body1" fontWeight="bold" gutterBottom>
            Parsed JSON Response:
          </Typography>
          <JSONPretty data={response} mainStyle="font-size: 14px;" keyStyle="font-weight: bold; color: #d63384;" stringStyle="color: #0d6efd;" />
        </Paper>
      )}

      {aiMessage && !response && (
        <Paper elevation={3} sx={{ maxWidth: 700, mt: 3, padding: 2, backgroundColor: "#e3f2fd" }}>
          <Typography variant="body1" fontWeight="bold" gutterBottom>
            AI Response:
          </Typography>
          <Typography variant="body2" color="textPrimary">
            {aiMessage}
          </Typography>
        </Paper>
      )}
    </Box>
  );
}