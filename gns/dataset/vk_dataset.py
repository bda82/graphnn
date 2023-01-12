import sys
import os
import numpy as np
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset
from gns.crawlers.vk.app.models import Group, GroupTagPivot, Tag, User, UserGroupPivot, UserUserPivot
from peewee import fn
import glob


class VkDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @property
    # def path(self):
    #     return os.path.dirname(__file__) + '/datasets'

    def download(self):
        data = ...  # Download from somewhere

        # Create the directory
        os.mkdir(self.path)

        # Get all tags
        tags = Tag.select(Tag.id, Tag.name).dicts()
        tags_count = len(tags)

        # Get all users with tags
        users = User.select().order_by(User.id)

        # Generate array of graphs
        output = []
        for user in users:

            # Get tags related to user
            user_tags = (
                User
                .select(
                    Tag.id.alias('tag_id'),
                    fn.Count(Tag.id).alias('tag_count')
                )
                .join(UserGroupPivot, on=(UserGroupPivot.user_id == User.id))
                .join(Group, on=(Group.id == UserGroupPivot.group_id))
                .join(GroupTagPivot, on=(GroupTagPivot.group_id == Group.id))
                .join(Tag, on=(GroupTagPivot.tag_id == Tag.id))
                .where(
                    User.id == user.id,
                )
                .group_by(Tag.id)
                .dicts()
            )

            # Get list of all friends
            friends = (
                UserUserPivot
                .select(
                    # UserUserPivot.from_id.alias('user_id'),
                    UserUserPivot.to_id.alias('friend_id'),
                    # Tag.id.alias('tag_id'),
                )
                .distinct()
                # .join(UserGroupPivot, on=(UserUserPivot.to_id == UserGroupPivot.user_id))
                # .join(Group, on=(Group.id == UserGroupPivot.group_id))
                # .join(GroupTagPivot, on=(Group.id == GroupTagPivot.group_id))
                # .join(Tag, on=(Tag.id == GroupTagPivot.tag_id))
                .where(UserUserPivot.from_id == user.id)
                .dicts()
            )
            friends_count = len(friends)

            # Skip users with small amount of friends
            if friends_count < 5:
                continue

            # Build output matrix
            matrix = [[0 for x in range(tags_count)] for y in range(friends_count + 1)]

            # First line is about current user
            for user_tag in user_tags:
                for x in range(tags_count):
                    tag = tags[x]
                    if user_tag['tag_id'] == tag['id']:
                        matrix[0][x] += 1

            # Get list of all tags related to each fiend
            for y in range(friends_count):
                friend = friends[y]
                user_id = user.id
                friend_id = friend['friend_id']
                # print(friend_id)

                # Get tags related to friend
                friend_tags = (
                    User
                    .select(
                        # User.id.alias('user_id'),
                        Tag.id.alias('tag_id'),
                        fn.Count(Tag.id).alias('tag_count')
                    )
                    .join(UserGroupPivot, on=(UserGroupPivot.user_id == User.id))
                    .join(Group, on=(Group.id == UserGroupPivot.group_id))
                    .join(GroupTagPivot, on=(GroupTagPivot.group_id == Group.id))
                    .join(Tag, on=(GroupTagPivot.tag_id == Tag.id))
                    .where(
                        User.id == friend_id,
                    )
                    .group_by(Tag.id)
                    .dicts()
                )

                # Skip friends without tags
                if len(friend_tags) < 1:
                    continue

                # Map tags by friends to matrix
                for friend_tag in friend_tags:
                    for x in range(tags_count):
                        tag = tags[x]
                        if friend_tag['tag_id'] == tag['id']:
                            matrix[y][x] += 1

            # Generate array from matrix
            x = np.array(matrix)

            # Save graph of user
            filename = os.path.join(self.path, f'graph_{user.id}')
            np.savez(
                filename,
                x=x,
                # a=a, #todo
            )

    def read(self):
        files = glob.glob(self.path + "/graph_*.npz")
        output = []
        for file in files:
            data = np.load(file)
            output.append(
                Graph(x=data['x'])
            )

        # Return list of graphs
        return 