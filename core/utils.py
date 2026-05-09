import time

# Dictionary to hold the last detected timestamp of product IDs
# Format: { product_id: timestamp }
cooldown_cache = {}

def check_cooldown(product_id, cooldown_seconds=3.0):
    """
    Returns True if the item is ALLOWED to be added.
    Returns False if it is blocked by active cooldown.
    """
    current_time = time.time()
    last_time = cooldown_cache.get(product_id)
    
    if last_time and (current_time - last_time < cooldown_seconds):
        return False
        
    # Update cache
    cooldown_cache[product_id] = current_time
    return True
