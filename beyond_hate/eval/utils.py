import base64
from io import BytesIO

from beyond_hate.train.utils import resize_and_pad


def pil_to_base64(image, img_format="PNG"):
    """Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        img_format: Image format (default: PNG)
    
    Returns:
        Base64 encoded string of the image
    """
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def get_reasoning_and_output(response):
    """Extract reasoning and output text from GPT response
    
    Args:
        response: OpenAI API response object
    
    Returns:
        Tuple of (reasoning, output_text)
    """
    reasoning = None
    output_text = None

    for item in response.output:
        if item.type == "reasoning":
            reasoning = "\n".join(item.summary) if item.summary else None
        elif item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    output_text = block.text

    return reasoning, output_text


def create_conversation(text, image, system_text, user_text, img_size, img_color_padding):
    """Create conversation format for GPT API call
    
    Args:
        text: Meme text content
        image: PIL Image object
        system_text: System prompt
        user_text: User prompt template
        img_size: Target image size as (width, height)
        img_color_padding: RGB color for padding as (R, G, B)
    
    Returns:
        Conversation list formatted for OpenAI API
    """
    if img_size:
        image = resize_and_pad(image, target_size=img_size, color=img_color_padding)
    
    img_b64 = pil_to_base64(image)
    img_url = f"data:image/png;base64,{img_b64}"
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_text}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text.format(text)},
                {"type": "input_image", "image_url": img_url},
            ]
        }
    ]
    return conversation
