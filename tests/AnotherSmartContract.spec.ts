import { Blockchain, SandboxContract, TreasuryContract } from '@ton/sandbox';
import { Cell, toNano } from '@ton/core';
import { AnotherSmartContract } from '../wrappers/AnotherSmartContract';
import '@ton/test-utils';
import { compile } from '@ton/blueprint';

describe('AnotherSmartContract', () => {
    let code: Cell;

    beforeAll(async () => {
        code = await compile('AnotherSmartContract');
    });

    let blockchain: Blockchain;
    let deployer: SandboxContract<TreasuryContract>;
    let anotherSmartContract: SandboxContract<AnotherSmartContract>;

    beforeEach(async () => {
        blockchain = await Blockchain.create();

        anotherSmartContract = blockchain.openContract(AnotherSmartContract.createFromConfig({}, code));

        deployer = await blockchain.treasury('deployer');

        const deployResult = await anotherSmartContract.sendDeploy(deployer.getSender(), toNano('0.05'));

        expect(deployResult.transactions).toHaveTransaction({
            from: deployer.address,
            to: anotherSmartContract.address,
            deploy: true,
            success: true,
        });
    });

    it('should deploy', async () => {
        // the check is done inside beforeEach
        // blockchain and anotherSmartContract are ready to use
    });
});
