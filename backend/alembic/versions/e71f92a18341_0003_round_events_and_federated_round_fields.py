"""0003_round_events_and_federated_round_fields

Revision ID: e71f92a18341
Revises: d4a7f29b3c10
Create Date: 2026-10-13 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e71f92a18341'
down_revision: Union[str, Sequence[str], None] = 'd4a7f29b3c10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add model_family and min_participants to federated_rounds
    with op.batch_alter_table('federated_rounds', schema=None) as batch_op:
        batch_op.add_column(sa.Column('model_family', sa.String(length=32), nullable=False, server_default='resnet18'))
        batch_op.add_column(sa.Column('min_participants', sa.Integer(), nullable=False, server_default='2'))

    # 2. Create round_events table
    op.create_table(
        'round_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('round_number', sa.Integer(), nullable=False),
        sa.Column('round_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('stage', sa.String(length=64), nullable=True),
        sa.Column('hospital_id', sa.Integer(), nullable=True),
        sa.Column('hospital_code', sa.String(length=32), nullable=True),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['hospital_id'], ['hospitals.id']),
        sa.ForeignKeyConstraint(['round_id'], ['federated_rounds.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('round_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_round_events_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_round_events_round_number'), ['round_number'], unique=False)
        batch_op.create_index(batch_op.f('ix_round_events_round_id'), ['round_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_round_events_event_type'), ['event_type'], unique=False)
        batch_op.create_index(batch_op.f('ix_round_events_hospital_id'), ['hospital_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_round_events_hospital_code'), ['hospital_code'], unique=False)


def downgrade() -> None:
    op.drop_table('round_events')
    with op.batch_alter_table('federated_rounds', schema=None) as batch_op:
        batch_op.drop_column('min_participants')
        batch_op.drop_column('model_family')
