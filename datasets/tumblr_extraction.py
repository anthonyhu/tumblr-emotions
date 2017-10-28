import pytumblr
import numpy as np

def extract_tumblr_posts(client, nb_requests, search_query, before, delta_limit):
    """Extract Tumblr posts with a given emotion.
    
    Parameters:
        client: Authenticated Tumblr client with the pytumblr package.
        nb_requests: Number of API request.
        search_query: Emotion to search for.
        before: A timestamp to search for posts before that value.
        delta_limit: Maximum difference of timestamp between two queries.
        
    Returns:
        posts: List of Tumblr posts. 
    """ 
    
    posts = []
    for i in range(nb_requests):
        tagged = client.tagged(search_query, filter='text', before=before)
        for elt in tagged:
            timestamp = elt['timestamp']
            if (abs(timestamp - before) < delta_limit):
                before = timestamp

                current_post = []
                current_post.append(elt['id'])
                current_post.append(elt['post_url'])
                elt_type = elt['type']
                current_post.append(elt_type)
                current_post.append(timestamp)
                current_post.append(elt['date'])
                current_post.append(elt['tags'])
                current_post.append(elt['liked'])
                current_post.append(elt['note_count'])

                if (elt_type == 'photo'):
                    # Only take the first image
                    current_post.append(elt['photos'][0]['original_size']['url'])
                    current_post.append(elt['caption'].replace('\n',' ').replace('\r',' '))
                    current_post.append(search_query)
                    posts.append(current_post)
                elif (elt_type == 'text'):
                    current_post.append(np.nan)
                    current_post.append(elt['body'].replace('\n',' ').replace('\r',' '))
                    current_post.append(search_query)
                    posts.append(current_post)
    return posts