"""0002_patient_pii_encryption_and_blind_indexes

Revision ID: d4a7f29b3c10
Revises: c885a98c1195
Create Date: 2026-10-12 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4a7f29b3c10'
down_revision: Union[str, Sequence[str], None] = 'c885a98c1195'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.alter_column('name', existing_type=sa.String(length=255), type_=sa.String(length=1024))
        batch_op.alter_column('phone', existing_type=sa.String(length=64), type_=sa.String(length=512))
        batch_op.add_column(sa.Column('email', sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column('name_bidx', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('phone_bidx', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('email_bidx', sa.String(length=64), nullable=True))
        batch_op.create_index(batch_op.f('ix_patients_name_bidx'), ['name_bidx'], unique=False)
        batch_op.create_index(batch_op.f('ix_patients_phone_bidx'), ['phone_bidx'], unique=False)
        batch_op.create_index(batch_op.f('ix_patients_email_bidx'), ['email_bidx'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_patients_email_bidx'))
        batch_op.drop_index(batch_op.f('ix_patients_phone_bidx'))
        batch_op.drop_index(batch_op.f('ix_patients_name_bidx'))
        batch_op.drop_column('email_bidx')
        batch_op.drop_column('phone_bidx')
        batch_op.drop_column('name_bidx')
        batch_op.drop_column('email')
        batch_op.alter_column('phone', existing_type=sa.String(length=512), type_=sa.String(length=64))
        batch_op.alter_column('name', existing_type=sa.String(length=1024), type_=sa.String(length=255))
