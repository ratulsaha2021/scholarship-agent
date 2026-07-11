"""Email sender module - handles SMTP email sending."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

CONFIG_DIR = Path(__file__).parent.parent / "config"

@dataclass
class EmailConfig:
    smtp_server: str = ""
    smtp_port: int = 587
    email_address: str = ""
    password: str = ""
    use_tls: bool = True
    
    @classmethod
    def load(cls) -> "EmailConfig":
        config_file = CONFIG_DIR / "email_config.json"
        if config_file.exists():
            import json
            with open(config_file, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()
    
    def save(self):
        import json
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_DIR / "email_config.json", "w") as f:
            json.dump({
                "smtp_server": self.smtp_server,
                "smtp_port": self.smtp_port,
                "email_address": self.email_address,
                "password": self.password,
                "use_tls": self.use_tls
            }, f, indent=2)

class EmailSender:
    """Sends emails via SMTP."""
    
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or EmailConfig.load()
    
    def is_configured(self) -> bool:
        """Check if email is configured."""
        return bool(self.config.smtp_server and self.config.email_address and self.config.password)
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_name: str = "",
        reply_to: Optional[str] = None
    ) -> Dict:
        """Send an email."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Email not configured. Go to Settings to set up SMTP."
            }
        
        try:
            msg = MIMEMultipart()
            msg['From'] = f"{from_name} <{self.config.email_address}>" if from_name else self.config.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            
            if reply_to:
                msg['Reply-To'] = reply_to
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.email_address, self.config.password)
                server.send_message(msg)
            
            return {
                "success": True,
                "message": f"Email sent to {to_email}",
                "to": to_email,
                "subject": subject
            }
        
        except smtplib.SMTPAuthenticationError:
            return {
                "success": False,
                "error": "Authentication failed. Check your email and password."
            }
        except smtplib.SMTPException as e:
            return {
                "success": False,
                "error": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to send email: {str(e)}"
            }
    
    def test_connection(self) -> Dict:
        """Test SMTP connection."""
        if not self.is_configured():
            return {"success": False, "error": "Email not configured"}
        
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.email_address, self.config.password)
            return {"success": True, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "error": str(e)}

def get_gmail_config() -> EmailConfig:
    """Get Gmail SMTP config."""
    return EmailConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        use_tls=True
    )

def get_outlook_config() -> EmailConfig:
    """Get Outlook SMTP config."""
    return EmailConfig(
        smtp_server="smtp-mail.outlook.com",
        smtp_port=587,
        use_tls=True
    )
