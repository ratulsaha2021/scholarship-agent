"""Main agent orchestrator - ties all modules together."""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .config import AgentConfig
from .resource_loader import UserResources, create_sample_resources
from .humanizer import Humanizer
from .discovery import OpportunityDiscovery, Opportunity, create_sample_targets
from .writer import EmailWriter, GeneratedEmail

console = Console()

class ScholarshipAgent:
    """Main agent for automated scholarship applications."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig.load()
        self.resources = UserResources.load()
        self.humanizer = Humanizer(level=self.config.humanization.level)
        self.discovery = OpportunityDiscovery(delay=self.config.discovery.delay_between_requests)
        self.writer = EmailWriter(self.resources, self.humanizer)
        
        self.generated_emails: List[GeneratedEmail] = []
        self.opportunities: List[Opportunity] = []
    
    def show_welcome(self):
        """Display welcome message and status."""
        console.print(Panel.fit(
            "[bold blue]Scholarship & PhD Application Agent[/bold blue]\n"
            "[dim]Automated applications with humanized writing[/dim]",
            border_style="blue"
        ))
        
        if not self.resources.name:
            console.print("[yellow]No user resources found. Running setup...[/yellow]")
            self.setup_resources()
        
        console.print(f"\n[green]Loaded resources for: {self.resources.name}[/green]")
        console.print(f"[dim]Research interests: {', '.join(self.resources.research_interests[:3])}...[/dim]")
    
    def setup_resources(self):
        """Interactive setup for user resources."""
        console.print("\n[bold]First-time Setup[/bold]")
        console.print("Let's set up your profile and resources.\n")
        
        name = Prompt.ask("Your full name")
        email = Prompt.ask("Your email")
        
        research = Prompt.ask("Top 3 research interests (comma-separated)")
        research_list = [r.strip() for r in research.split(",")]
        
        resources_dir = self.config.resources_dir
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        user_data = {
            "name": name,
            "email": email,
            "research_interests": research_list,
            "education": [],
            "experience": [],
            "publications": [],
            "awards": []
        }
        
        with open(resources_dir / "user_data.json", "w") as f:
            json.dump(user_data, f, indent=2)
        
        console.print(f"\n[green]Profile saved to {resources_dir / 'user_data.json'}[/green]")
        console.print("[yellow]Please add your CV (cv.pdf or cv.docx) to the resources folder.[/yellow]")
        console.print(f"[dim]Resources folder: {resources_dir}[/dim]\n")
        
        self.resources = UserResources.load(resources_dir)
    
    def discover_opportunities(self, query: str = "") -> List[Opportunity]:
        """Discover available opportunities."""
        console.print("\n[bold cyan]Discovering Opportunities...[/bold cyan]")
        
        manual_targets = self.discovery.load_manual_targets()
        console.print(f"[dim]Found {len(manual_targets)} manual targets[/dim]")
        
        all_opportunities = list(manual_targets)
        
        if self.config.discovery.scrape_enabled and query:
            with console.status("[green]Searching online...[/green]"):
                online_results = self.discovery.search_academic_positions(query)
                all_opportunities.extend(online_results)
                console.print(f"[dim]Found {len(online_results)} online results[/dim]")
        
        self.opportunities = all_opportunities
        return all_opportunities
    
    def display_opportunities(self, opportunities: List[Opportunity]):
        """Display discovered opportunities in a table."""
        if not opportunities:
            console.print("[yellow]No opportunities found.[/yellow]")
            return
        
        table = Table(title="Discovered Opportunities")
        table.add_column("Type", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Institution", style="green")
        table.add_column("Deadline", style="yellow")
        
        for i, opp in enumerate(opportunities, 1):
            table.add_row(
                opp.type,
                opp.title[:50],
                opp.institution,
                opp.deadline or "N/A"
            )
        
        console.print(table)
    
    def generate_professor_email(
        self,
        professor_name: str,
        email: str,
        research_topic: str,
        custom_message: Optional[str] = None
    ) -> GeneratedEmail:
        """Generate and display a professor email."""
        console.print(f"\n[bold cyan]Generating email to Professor {professor_name}...[/bold cyan]")
        
        with console.status("[green]Writing humanized email...[/green]"):
            generated = self.writer.write_professor_email(
                professor_name, email, research_topic, custom_message=custom_message
            )
        
        self._display_generated_email(generated)
        self.generated_emails.append(generated)
        
        return generated
    
    def generate_scholarship_application(
        self,
        opportunity: Opportunity,
        additional_info: Optional[str] = None
    ) -> GeneratedEmail:
        """Generate and display a scholarship application."""
        console.print(f"\n[bold cyan]Generating application for: {opportunity.title}...[/bold cyan]")
        
        with console.status("[green]Writing humanized application...[/green]"):
            generated = self.writer.write_scholarship_application(opportunity, additional_info)
        
        self._display_generated_email(generated)
        self.generated_emails.append(generated)
        
        return generated
    
    def _display_generated_email(self, email: GeneratedEmail):
        """Display generated email with review."""
        review = self.writer.review_email(email)
        
        console.print(Panel(
            f"[bold]Subject:[/bold] {email.subject}\n"
            f"[bold]To:[/bold] {email.to_email}\n\n"
            f"{email.body}",
            title="Generated Email",
            border_style="green"
        ))
        
        console.print(Panel(
            f"[bold]Word Count:[/bold] {review['word_count']}\n"
            f"[bold]Humanization Score:[/bold] {review['humanization_score']:.2f}\n"
            f"[bold]AI Patterns Found:[/bold] {review['ai_patterns_found']}\n"
            f"[bold]Changes Made:[/bold] {', '.join(review['changes_made']) if review['changes_made'] else 'None'}",
            title="Email Review",
            border_style="yellow"
        ))
    
    def interactive_mode(self):
        """Run interactive mode."""
        self.show_welcome()
        
        while True:
            console.print("\n[bold]Options:[/bold]")
            console.print("1. Discover opportunities")
            console.print("2. Write professor email")
            console.print("3. Apply to scholarship")
            console.print("4. View generated emails")
            console.print("5. Save all emails")
            console.print("6. Setup/Resources")
            console.print("7. Exit")
            
            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7"])
            
            if choice == "1":
                query = Prompt.ask("Search query (or press Enter for manual targets only)")
                opps = self.discover_opportunities(query)
                self.display_opportunities(opps)
            
            elif choice == "2":
                prof_name = Prompt.ask("Professor name")
                prof_email = Prompt.ask("Professor email")
                topic = Prompt.ask("Their research topic you're interested in")
                custom = Prompt.ask("Any additional message? (optional)", default="")
                self.generate_professor_email(prof_name, prof_email, topic, custom or None)
            
            elif choice == "3":
                if not self.opportunities:
                    console.print("[yellow]No opportunities loaded. Run discovery first.[/yellow]")
                    continue
                
                self.display_opportunities(self.opportunities)
                idx = int(Prompt.ask("Select opportunity number")) - 1
                
                if 0 <= idx < len(self.opportunities):
                    additional = Prompt.ask("Any additional info to include? (optional)", default="")
                    self.generate_scholarship_application(
                        self.opportunities[idx], additional or None
                    )
            
            elif choice == "4":
                if not self.generated_emails:
                    console.print("[yellow]No emails generated yet.[/yellow]")
                    continue
                
                for i, email in enumerate(self.generated_emails, 1):
                    console.print(f"\n[bold]{i}. {email.subject}[/bold] -> {email.to_email}")
            
            elif choice == "5":
                self.save_all_emails()
            
            elif choice == "6":
                create_sample_resources()
                self.resources = UserResources.load()
                console.print("[green]Resources reloaded.[/green]")
            
            elif choice == "7":
                console.print("[dim]Goodbye![/dim]")
                break
    
    def auto_mode(self, targets_file: Optional[Path] = None):
        """Run automatic mode - apply to all targets."""
        self.show_welcome()
        
        console.print("\n[bold cyan]Auto Mode: Applying to all targets...[/bold cyan]")
        
        opportunities = self.discover_opportunities()
        self.display_opportunities(opportunities)
        
        if not opportunities:
            console.print("[red]No opportunities found. Add targets to resources/targets.json[/red]")
            return
        
        confirm = Confirm.ask(f"Apply to {len(opportunities)} opportunities?")
        if not confirm:
            return
        
        for i, opp in enumerate(opportunities, 1):
            console.print(f"\n[bold]({i}/{len(opportunities)}) {opp.title}[/bold]")
            
            if opp.type == "professor" and opp.professor_email:
                generated = self.writer.write_professor_email(
                    opp.title.split(" in ")[-1] if " in " in opp.title else opp.title,
                    opp.professor_email,
                    opp.description
                )
            else:
                generated = self.writer.write_scholarship_application(opp)
            
            self._display_generated_email(generated)
            self.generated_emails.append(generated)
            
            if i < len(opportunities):
                console.print("[dim]Waiting 2 seconds before next...[/dim]")
                import time
                time.sleep(2)
        
        self.save_all_emails()
        console.print(f"\n[green]Generated {len(self.generated_emails)} emails![/green]")
    
    def save_all_emails(self):
        """Save all generated emails to files."""
        if not self.generated_emails:
            console.print("[yellow]No emails to save.[/yellow]")
            return
        
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, email in enumerate(self.generated_emails):
            filename = f"email_{i+1}_{email.to_email.split('@')[0]}.txt"
            filepath = output_dir / filename
            
            content = f"Subject: {email.subject}\n"
            content += f"To: {email.to_email}\n"
            content += f"Humanization Score: {email.humanization_result.confidence_score:.2f}\n"
            content += "=" * 60 + "\n\n"
            content += email.body
            
            with open(filepath, "w") as f:
                f.write(content)
        
        console.print(f"[green]Saved {len(self.generated_emails)} emails to {output_dir}[/green]")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scholarship & PhD Application Agent")
    parser.add_argument("--auto", action="store_true", help="Run in auto mode")
    parser.add_argument("--professor", type=str, help="Generate email for specific professor")
    parser.add_argument("--email", type=str, help="Professor email (with --professor)")
    parser.add_argument("--topic", type=str, help="Research topic (with --professor)")
    parser.add_argument("--setup", action="store_true", help="Run initial setup")
    
    args = parser.parse_args()
    
    agent = ScholarshipAgent()
    
    if args.setup:
        create_sample_resources()
        create_sample_targets()
        console.print("[green]Setup complete! Edit the resource files with your information.[/green]")
        return
    
    if args.auto:
        agent.auto_mode()
    elif args.professor:
        if not args.email or not args.topic:
            console.print("[red]--email and --topic required with --professor[/red]")
            return
        agent.show_welcome()
        agent.generate_professor_email(args.professor, args.email, args.topic)
    else:
        agent.interactive_mode()

if __name__ == "__main__":
    main()
