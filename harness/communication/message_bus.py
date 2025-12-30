"""
Inter-Agent Communication Protocol
Enables agents to communicate, collaborate, and coordinate.

Key features:
- Point-to-point messaging
- Broadcast/multicast messaging
- Topic-based pub/sub
- Request/response patterns
- Conversation threading
- Message persistence
- Priority queuing

Design inspired by:
- Manus's multi-agent coordination
- Actor model patterns
- Message queue systems (RabbitMQ, Redis)
"""

import json
import uuid
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import hashlib


class MessageType(Enum):
    """Types of inter-agent messages"""
    # Direct communication
    DIRECT = "direct"           # Point-to-point
    BROADCAST = "broadcast"     # To all agents
    MULTICAST = "multicast"     # To specific group

    # Request/response
    REQUEST = "request"
    RESPONSE = "response"

    # Pub/sub
    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Coordination
    TASK_ASSIGN = "task_assign"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"

    # System
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class Message:
    """
    Universal message format for inter-agent communication.
    Works with both tool-using and non-tool-using LLMs.
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # Time to live
    correlation_id: Optional[str] = None  # For request/response
    conversation_id: Optional[str] = None  # Thread tracking
    topic: Optional[str] = None  # For pub/sub
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_ack: bool = False
    encrypted: bool = False

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "correlation_id": self.correlation_id,
            "conversation_id": self.conversation_id,
            "topic": self.topic,
            "metadata": self.metadata
        }

    def to_text_format(self) -> str:
        """
        Convert to text format for non-tool-using LLMs.
        This is how messages appear in an agent's context.
        """
        lines = [
            f"[MESSAGE from {self.sender_id}]",
            f"ID: {self.message_id}",
            f"Type: {self.message_type.value}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if self.topic:
            lines.append(f"Topic: {self.topic}")

        if self.conversation_id:
            lines.append(f"Thread: {self.conversation_id}")

        lines.append("")

        if isinstance(self.content, str):
            lines.append(self.content)
        else:
            lines.append(json.dumps(self.content, indent=2))

        lines.append("[/MESSAGE]")
        return '\n'.join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data["type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            content=data["content"],
            priority=MessagePriority(data.get("priority", 1)),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            ttl_seconds=data.get("ttl_seconds", 3600),
            correlation_id=data.get("correlation_id"),
            conversation_id=data.get("conversation_id"),
            topic=data.get("topic"),
            metadata=data.get("metadata", {})
        )

    def is_expired(self) -> bool:
        expiry = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry


@dataclass
class Conversation:
    """Represents a message thread between agents"""
    conversation_id: str
    participants: Set[str]
    created_at: datetime
    last_activity: datetime
    messages: List[str]  # message_ids
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageHandler(ABC):
    """Abstract handler for processing messages"""

    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        pass

    @abstractmethod
    def handle(self, message: Message) -> Optional[Message]:
        """Handle message, optionally return response"""
        pass


class AgentMailbox:
    """
    Mailbox for a single agent.
    Manages incoming/outgoing messages and subscriptions.
    """

    def __init__(self, agent_id: str, max_queue_size: int = 1000):
        self.agent_id = agent_id
        self.inbox: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self.outbox: queue.Queue = queue.Queue()
        self.subscriptions: Set[str] = set()  # topic subscriptions
        self.handlers: List[MessageHandler] = []
        self.pending_responses: Dict[str, threading.Event] = {}
        self.response_store: Dict[str, Message] = {}
        self._lock = threading.Lock()

    def receive(self, message: Message):
        """Receive a message into inbox"""
        if message.is_expired():
            return

        # Priority queue uses (priority, timestamp, message) tuple
        # Lower priority number = higher priority, so we negate
        priority_key = -message.priority.value
        self.inbox.put((priority_key, message.timestamp.timestamp(), message))

    def send(self, message: Message):
        """Queue message for sending"""
        message.sender_id = self.agent_id
        self.outbox.put(message)

    def get_next_message(self, timeout: float = None) -> Optional[Message]:
        """Get next message from inbox"""
        try:
            _, _, message = self.inbox.get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def peek_messages(self, limit: int = 10) -> List[Message]:
        """Peek at messages without removing them"""
        messages = []
        temp = []

        while len(messages) < limit:
            try:
                item = self.inbox.get_nowait()
                temp.append(item)
                messages.append(item[2])
            except queue.Empty:
                break

        # Put them back
        for item in temp:
            self.inbox.put(item)

        return messages

    def subscribe(self, topic: str):
        """Subscribe to a topic"""
        self.subscriptions.add(topic)

    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        self.subscriptions.discard(topic)

    def is_subscribed(self, topic: str) -> bool:
        """Check if subscribed to topic"""
        return topic in self.subscriptions

    def add_handler(self, handler: MessageHandler):
        """Add a message handler"""
        self.handlers.append(handler)

    def process_messages(self, max_messages: int = 10) -> List[Message]:
        """Process pending messages using handlers"""
        responses = []

        for _ in range(max_messages):
            message = self.get_next_message(timeout=0.1)
            if not message:
                break

            for handler in self.handlers:
                if handler.can_handle(message):
                    response = handler.handle(message)
                    if response:
                        responses.append(response)
                    break

        return responses

    def wait_for_response(
        self,
        correlation_id: str,
        timeout: float = 30
    ) -> Optional[Message]:
        """Wait for a response to a request"""
        event = threading.Event()
        self.pending_responses[correlation_id] = event

        if event.wait(timeout):
            return self.response_store.pop(correlation_id, None)

        return None


class MessageBus:
    """
    Central message bus for inter-agent communication.
    Routes messages between agents.
    """

    def __init__(self):
        self.mailboxes: Dict[str, AgentMailbox] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.message_log: List[Message] = []
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_agent(self, agent_id: str) -> AgentMailbox:
        """Register an agent and create mailbox"""
        with self._lock:
            if agent_id not in self.mailboxes:
                self.mailboxes[agent_id] = AgentMailbox(agent_id)
            return self.mailboxes[agent_id]

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            if agent_id in self.mailboxes:
                # Remove from all topic subscriptions
                for subscribers in self.topic_subscribers.values():
                    subscribers.discard(agent_id)
                del self.mailboxes[agent_id]

    def get_mailbox(self, agent_id: str) -> Optional[AgentMailbox]:
        """Get an agent's mailbox"""
        return self.mailboxes.get(agent_id)

    def send(self, message: Message) -> bool:
        """Send a message through the bus"""
        self.message_log.append(message)

        if message.message_type == MessageType.BROADCAST:
            return self._broadcast(message)

        elif message.message_type == MessageType.MULTICAST:
            recipients = message.metadata.get("recipients", [])
            return self._multicast(message, recipients)

        elif message.message_type == MessageType.PUBLISH:
            return self._publish(message)

        elif message.message_type == MessageType.SUBSCRIBE:
            return self._subscribe(message.sender_id, message.topic)

        elif message.message_type == MessageType.UNSUBSCRIBE:
            return self._unsubscribe(message.sender_id, message.topic)

        elif message.recipient_id:
            return self._direct_send(message)

        return False

    def _direct_send(self, message: Message) -> bool:
        """Send direct message to recipient"""
        mailbox = self.mailboxes.get(message.recipient_id)
        if mailbox:
            mailbox.receive(message)

            # Handle response correlation
            if message.message_type == MessageType.RESPONSE and message.correlation_id:
                sender_mailbox = self.mailboxes.get(message.sender_id)
                if sender_mailbox and message.correlation_id in sender_mailbox.pending_responses:
                    sender_mailbox.response_store[message.correlation_id] = message
                    sender_mailbox.pending_responses[message.correlation_id].set()

            return True
        return False

    def _broadcast(self, message: Message) -> bool:
        """Broadcast message to all agents"""
        for agent_id, mailbox in self.mailboxes.items():
            if agent_id != message.sender_id:
                mailbox.receive(message)
        return True

    def _multicast(self, message: Message, recipients: List[str]) -> bool:
        """Send to multiple specific recipients"""
        success = True
        for recipient_id in recipients:
            if recipient_id in self.mailboxes and recipient_id != message.sender_id:
                self.mailboxes[recipient_id].receive(message)
            else:
                success = False
        return success

    def _publish(self, message: Message) -> bool:
        """Publish message to topic subscribers"""
        if not message.topic:
            return False

        subscribers = self.topic_subscribers.get(message.topic, set())
        for subscriber_id in subscribers:
            if subscriber_id != message.sender_id:
                mailbox = self.mailboxes.get(subscriber_id)
                if mailbox:
                    mailbox.receive(message)
        return True

    def _subscribe(self, agent_id: str, topic: str) -> bool:
        """Subscribe agent to topic"""
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(agent_id)

        mailbox = self.mailboxes.get(agent_id)
        if mailbox:
            mailbox.subscribe(topic)
        return True

    def _unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe agent from topic"""
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(agent_id)

        mailbox = self.mailboxes.get(agent_id)
        if mailbox:
            mailbox.unsubscribe(topic)
        return True

    def request(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        timeout: float = 30
    ) -> Optional[Message]:
        """Send request and wait for response"""
        correlation_id = str(uuid.uuid4())

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            correlation_id=correlation_id,
            requires_ack=True
        )

        sender_mailbox = self.mailboxes.get(sender_id)
        if not sender_mailbox:
            return None

        self.send(message)
        return sender_mailbox.wait_for_response(correlation_id, timeout)

    def respond(
        self,
        original_message: Message,
        content: Any,
        sender_id: str
    ) -> bool:
        """Send response to a request"""
        response = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            content=content,
            correlation_id=original_message.correlation_id,
            conversation_id=original_message.conversation_id
        )
        return self.send(response)

    def start_conversation(
        self,
        initiator_id: str,
        participant_ids: List[str],
        topic: str = None
    ) -> Conversation:
        """Start a new conversation thread"""
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            participants=set([initiator_id] + participant_ids),
            created_at=datetime.now(),
            last_activity=datetime.now(),
            messages=[],
            topic=topic
        )
        self.conversations[conversation.conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)

    def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages in a conversation"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return []

        return [
            msg for msg in self.message_log
            if msg.conversation_id == conversation_id
        ]

    def get_topics(self) -> List[str]:
        """Get all active topics"""
        return list(self.topic_subscribers.keys())

    def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get subscribers for a topic"""
        return list(self.topic_subscribers.get(topic, set()))

    def cleanup_expired(self):
        """Remove expired messages from logs"""
        self.message_log = [
            msg for msg in self.message_log
            if not msg.is_expired()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            "registered_agents": len(self.mailboxes),
            "active_topics": len(self.topic_subscribers),
            "total_conversations": len(self.conversations),
            "message_log_size": len(self.message_log),
            "topic_details": {
                topic: len(subs)
                for topic, subs in self.topic_subscribers.items()
            }
        }


# Helper functions for creating common message types
def create_task_message(
    sender_id: str,
    recipient_id: str,
    task_description: str,
    task_params: Dict[str, Any] = None,
    priority: MessagePriority = MessagePriority.NORMAL
) -> Message:
    """Create a task assignment message"""
    return Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.TASK_ASSIGN,
        sender_id=sender_id,
        recipient_id=recipient_id,
        content={
            "description": task_description,
            "params": task_params or {}
        },
        priority=priority,
        requires_ack=True
    )


def create_broadcast_message(
    sender_id: str,
    content: Any,
    topic: str = None
) -> Message:
    """Create a broadcast message"""
    return Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.BROADCAST,
        sender_id=sender_id,
        recipient_id=None,
        content=content,
        topic=topic
    )


def create_publish_message(
    sender_id: str,
    topic: str,
    content: Any
) -> Message:
    """Create a topic publish message"""
    return Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.PUBLISH,
        sender_id=sender_id,
        recipient_id=None,
        content=content,
        topic=topic
    )


# Text-based message parsing for non-tool-using LLMs
class TextMessageParser:
    """
    Parse messages from text format for non-tool-using LLMs.

    Example formats:
    @agent-id message content here
    @broadcast message to everyone
    #topic message about this topic
    """

    PATTERNS = {
        'direct': r'^@(\S+)\s+(.+)$',
        'broadcast': r'^@broadcast\s+(.+)$',
        'topic': r'^#(\S+)\s+(.+)$',
        'reply': r'^>>\s*(\S+)\s+(.+)$',  # >> message_id reply content
    }

    def parse(self, text: str, sender_id: str) -> Optional[Message]:
        """Parse text into a Message"""
        import re

        text = text.strip()

        # Broadcast
        match = re.match(self.PATTERNS['broadcast'], text)
        if match:
            return create_broadcast_message(sender_id, match.group(1))

        # Topic publish
        match = re.match(self.PATTERNS['topic'], text)
        if match:
            return create_publish_message(sender_id, match.group(1), match.group(2))

        # Direct message
        match = re.match(self.PATTERNS['direct'], text)
        if match:
            return Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.DIRECT,
                sender_id=sender_id,
                recipient_id=match.group(1),
                content=match.group(2)
            )

        return None


# Format helpers for displaying messages to LLMs
def format_inbox_for_llm(mailbox: AgentMailbox, max_messages: int = 5) -> str:
    """Format mailbox contents for LLM context"""
    messages = mailbox.peek_messages(max_messages)

    if not messages:
        return "[No new messages]"

    lines = [f"[INBOX - {len(messages)} message(s)]", ""]

    for msg in messages:
        lines.append(msg.to_text_format())
        lines.append("")

    lines.append("[/INBOX]")
    return '\n'.join(lines)


def format_message_reply_options(message: Message) -> str:
    """Generate reply options for an LLM"""
    return f"""To reply to this message, use one of these formats:

```command:msg.send
to: {message.sender_id}
message: Your reply here
conversation_id: {message.conversation_id or message.message_id}
```

Or in natural language:
@{message.sender_id} Your reply here
"""
